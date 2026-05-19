from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.nn.functional as F

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _load_token_subset(path_str: str, key: str = "tokens") -> List[str]:
    """Load a token subset from json/csv/plain-text file."""
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get(key, [])
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
        return [str(token) for token in data if str(token)]
    if suffix == ".csv":
        df = pd.read_csv(path)
        if "token" not in df.columns:
            raise ValueError(f"CSV {path} must contain a 'token' column")
        return [str(token) for token in df["token"].dropna().astype(str).tolist()]

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _prepare_bev_labels_for_eval(
    agent: AbstractAgent,
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    helper = getattr(agent, "_prepare_bev_target_labels_for_gate", None)
    if callable(helper):
        try:
            labels = helper(pred_logits, target_labels)
            if torch.is_tensor(labels):
                return labels
        except Exception:
            pass

    labels = target_labels
    if labels.dim() == 4 and labels.shape[1] == pred_logits.shape[1]:
        labels = labels.argmax(dim=1)
    if labels.dim() == 3 and labels.shape[0] == pred_logits.shape[1]:
        labels = labels.argmax(dim=0, keepdim=True)
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    if labels.dim() != 3:
        return None
    if labels.shape[0] != pred_logits.shape[0]:
        if labels.shape[0] == 1:
            labels = labels.repeat(pred_logits.shape[0], 1, 1)
        else:
            return None
    return labels.to(pred_logits.device).long()


def _batchify_feature_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}
    for k, v in data.items():
        batched[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
    return batched


def _batchify_target_dict(
    data: Dict[str, Any], token_override: Optional[str] = None
) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            batched[k] = v.unsqueeze(0)
        elif isinstance(v, (str, bytes)):
            batched[k] = [v]
        else:
            batched[k] = v
    if token_override is not None:
        batched["token"] = [token_override]
    return batched


def _load_cached_tokens_per_log(
    cache_root: Path,
    log_names: Optional[List[str]] = None,
    tokens: Optional[List[str]] = None,
    max_scenes: Optional[int] = None,
) -> Dict[str, List[str]]:
    allowed_logs = set(log_names) if log_names is not None else None
    allowed_tokens = set(tokens) if tokens is not None else None
    tokens_per_log: Dict[str, List[str]] = {}
    total = 0
    for log_dir in sorted(cache_root.iterdir()):
        if not log_dir.is_dir():
            continue
        log_name = log_dir.name
        if allowed_logs is not None and log_name not in allowed_logs:
            continue
        token_list: List[str] = []
        for token_dir in sorted(log_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            token = token_dir.name
            if allowed_tokens is not None and token not in allowed_tokens:
                continue
            token_list.append(token)
            total += 1
            if max_scenes is not None and total >= int(max_scenes):
                break
        if token_list:
            tokens_per_log[log_name] = token_list
        if max_scenes is not None and total >= int(max_scenes):
            break
    return tokens_per_log


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    prefer_training_cache = bool(cfg.get("prefer_training_cache", False))
    training_cache_path = cfg.get("training_cache_path", None)
    record_hardcase_metrics = bool(cfg.get("record_hardcase_metrics", False))
    record_bev_loss_metrics = bool(cfg.get("record_bev_loss_metrics", False))
    record_attention_loss_metrics = bool(
        cfg.get("record_attention_loss_metrics", False)
    )
    record_mode_oracle_metrics = bool(cfg.get("record_mode_oracle_metrics", False))
    need_extra_eval_metrics = (
        record_bev_loss_metrics
        or record_attention_loss_metrics
        or record_mode_oracle_metrics
    )

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent_cfg = getattr(agent, "_config", None)
    if agent_cfg is not None:
        if record_hardcase_metrics and not bool(
            getattr(agent_cfg, "hardcase_record_enable", False)
        ):
            setattr(agent_cfg, "hardcase_record_enable", True)
            logger.info(
                "record_hardcase_metrics=True: enabling agent.config.hardcase_record_enable for this run."
            )
        if record_attention_loss_metrics:
            if not bool(getattr(agent_cfg, "pdm_score_output_infraction_details", False)):
                setattr(agent_cfg, "pdm_score_output_infraction_details", True)
                logger.info(
                    "record_attention_loss_metrics=True: enabling agent.config.pdm_score_output_infraction_details for this run."
                )
            if not bool(getattr(agent_cfg, "pdm_score_output_infraction_in_val", False)):
                setattr(agent_cfg, "pdm_score_output_infraction_in_val", True)
                logger.info(
                    "record_attention_loss_metrics=True: enabling agent.config.pdm_score_output_infraction_in_val for this run."
                )
        if (record_mode_oracle_metrics or record_attention_loss_metrics) and not bool(
            getattr(agent_cfg, "pdm_val_use_online_score", False)
        ):
            setattr(agent_cfg, "pdm_val_use_online_score", True)
            logger.info(
                "record_mode_oracle_metrics/record_attention_loss_metrics=True: "
                "enabling agent.config.pdm_val_use_online_score for this run."
            )

    feature_builders = []
    target_builders = []
    cache_dataset: Optional[CacheOnlyDataset] = None
    cache_tokens_available = set()

    if prefer_training_cache and training_cache_path:
        try:
            feature_builders = list(agent.get_feature_builders())
            target_builders = list(agent.get_target_builders())
            cache_dataset = CacheOnlyDataset(
                cache_path=str(training_cache_path),
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=log_names,
                tokens=tokens,
            )
            cache_tokens_available = set(cache_dataset.tokens)
            if len(cache_tokens_available) == 0:
                cache_dataset = None
        except Exception:
            logger.warning(
                "prefer_training_cache=True but failed to initialize CacheOnlyDataset. Falling back to raw scene loading.",
                exc_info=True,
            )
            cache_dataset = None
            cache_tokens_available = set()

    use_cache_for_primary_inference = (
        cache_dataset is not None and not bool(agent.requires_scene)
    )
    use_cache_for_extra_eval = cache_dataset is not None

    if need_extra_eval_metrics:
        try:
            if not feature_builders:
                feature_builders = list(agent.get_feature_builders())
            if len(feature_builders) == 0:
                logger.warning(
                    "Extra eval metrics requested but agent has no feature builders. Disabling BEV/attention/oracle metrics."
                )
                record_bev_loss_metrics = False
                record_attention_loss_metrics = False
                record_mode_oracle_metrics = False
                need_extra_eval_metrics = False
            elif record_bev_loss_metrics:
                if not target_builders:
                    target_builders = list(agent.get_target_builders())
                if len(target_builders) == 0:
                    logger.warning(
                        "record_bev_loss_metrics=True but agent has no target builders. Disabling BEV metrics."
                    )
                    record_bev_loss_metrics = False
        except Exception:
            logger.warning(
                "Extra eval metrics requested but failed to get feature/target builders. Disabling BEV/attention/oracle metrics."
            )
            record_bev_loss_metrics = False
            record_attention_loss_metrics = False
            record_mode_oracle_metrics = False
            need_extra_eval_metrics = False
        need_extra_eval_metrics = (
            record_bev_loss_metrics
            or record_attention_loss_metrics
            or record_mode_oracle_metrics
        )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    missing_cache_tokens = set(tokens) - cache_tokens_available
    need_scene_loader = bool(agent.requires_scene) or len(missing_cache_tokens) > 0
    extra_eval_warning_budget = 5
    scene_loader = None
    if need_scene_loader:
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_names
        scene_filter.tokens = tokens
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=agent.get_sensor_config(),
        )

    if scene_loader is not None:
        tokens_to_evaluate = [
            token
            for token in tokens
            if token in metric_cache_loader.tokens and token in scene_loader.tokens
        ]
    else:
        tokens_to_evaluate = [
            token
            for token in tokens
            if token in metric_cache_loader.tokens and token in cache_tokens_available
        ]
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            scene = None
            cached_features_raw: Optional[Dict[str, Any]] = None
            cached_targets_raw: Optional[Dict[str, Any]] = None
            eval_predictions: Optional[Dict[str, Any]] = None
            eval_targets_batched: Optional[Dict[str, Any]] = None

            use_cache_for_token = token in cache_tokens_available
            if use_cache_for_primary_inference and use_cache_for_token and cache_dataset is not None:
                try:
                    cached_features_raw, cached_targets_raw = cache_dataset._load_scene_with_token(token)
                    eval_features = _batchify_feature_dict(cached_features_raw)
                    with torch.no_grad():
                        try:
                            eval_predictions = agent.forward(eval_features)
                        except TypeError:
                            eval_predictions = agent.forward(eval_features, None)
                    hook = getattr(agent, "_on_inference_predictions", None)
                    if callable(hook):
                        try:
                            hook(eval_predictions)
                        except Exception:
                            pass
                    trajectory = Trajectory(
                        eval_predictions["trajectory"].squeeze(0).detach().cpu().numpy()
                    )
                    eval_targets_batched = _batchify_target_dict(
                        cached_targets_raw, token_override=token
                    )
                except Exception:
                    logger.debug(
                        "Failed to use training cache for token %s; falling back to raw scene loading.",
                        token,
                        exc_info=True,
                    )
                    cached_features_raw = None
                    cached_targets_raw = None
                    eval_predictions = None
                    eval_targets_batched = None

            if eval_predictions is None:
                assert scene_loader is not None
                agent_input = scene_loader.get_agent_input_from_token(token)
                if agent.requires_scene or (
                    record_bev_loss_metrics and not use_cache_for_token
                ):
                    scene = scene_loader.get_scene_from_token(token)
                if agent.requires_scene:
                    assert scene is not None
                    trajectory = agent.compute_trajectory(agent_input, scene)
                else:
                    trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))

            if need_extra_eval_metrics:
                try:
                    if eval_predictions is None:
                        if use_cache_for_extra_eval and use_cache_for_token and cache_dataset is not None:
                            if cached_features_raw is None or cached_targets_raw is None:
                                cached_features_raw, cached_targets_raw = cache_dataset._load_scene_with_token(token)
                            eval_features = _batchify_feature_dict(cached_features_raw)
                            eval_targets_batched = _batchify_target_dict(
                                cached_targets_raw, token_override=token
                            )
                        else:
                            assert scene_loader is not None
                            agent_input = scene_loader.get_agent_input_from_token(token)
                            eval_features = {}
                            for builder in feature_builders:
                                eval_features.update(builder.compute_features(agent_input))

                            eval_features = _batchify_feature_dict(eval_features)
                            eval_targets_batched = {"token": [token]}
                            if record_bev_loss_metrics:
                                if scene is None:
                                    scene = scene_loader.get_scene_from_token(token)
                                eval_targets: Dict[str, Any] = {}
                                for builder in target_builders:
                                    eval_targets.update(builder.compute_targets(scene))
                                for k, v in eval_targets.items():
                                    if torch.is_tensor(v):
                                        eval_targets_batched[k] = v.unsqueeze(0)
                                    elif isinstance(v, (str, bytes)):
                                        eval_targets_batched[k] = [v]
                                    else:
                                        eval_targets_batched[k] = v

                        with torch.no_grad():
                            try:
                                eval_predictions = agent.forward(
                                    eval_features, eval_targets_batched
                                )
                            except TypeError:
                                eval_predictions = agent.forward(eval_features)

                    if record_bev_loss_metrics:
                        pred_bev = eval_predictions.get("bev_semantic_map")
                        target_bev = eval_targets_batched.get("bev_semantic_map")
                        if torch.is_tensor(pred_bev) and torch.is_tensor(target_bev):
                            bev_labels = _prepare_bev_labels_for_eval(
                                agent=agent,
                                pred_logits=pred_bev,
                                target_labels=target_bev,
                            )
                            if torch.is_tensor(bev_labels):
                                bev_ce = F.cross_entropy(
                                    pred_bev.detach().float(),
                                    bev_labels,
                                    reduction="none",
                                )
                                bev_ce_per_sample = bev_ce.view(
                                    bev_ce.shape[0], -1
                                ).mean(dim=-1)
                                score_row["hc_bev_ce"] = float(
                                    bev_ce_per_sample.mean().item()
                                )

                    if record_mode_oracle_metrics or record_attention_loss_metrics:
                        pdm_metric_fn = getattr(agent, "compute_pdm_val_metrics", None)
                        if callable(pdm_metric_fn):
                            agent_cfg = getattr(agent, "_config", None)
                            prev_use_selected = None
                            prev_topk = None
                            if agent_cfg is not None:
                                prev_use_selected = getattr(
                                    agent_cfg,
                                    "pdm_val_score_use_selected_trajectory",
                                    None,
                                )
                                prev_topk = getattr(
                                    agent_cfg,
                                    "pdm_val_score_topk",
                                    None,
                                )
                                setattr(
                                    agent_cfg,
                                    "pdm_val_score_use_selected_trajectory",
                                    False,
                                )
                                setattr(agent_cfg, "pdm_val_score_topk", 0)
                            try:
                                oracle_metrics = pdm_metric_fn(
                                    eval_targets_batched,
                                    eval_predictions,
                                    metric_cache_path_override=str(
                                        cfg.metric_cache_path
                                    ),
                                )
                            finally:
                                if agent_cfg is not None and prev_use_selected is not None:
                                    setattr(
                                        agent_cfg,
                                        "pdm_val_score_use_selected_trajectory",
                                        prev_use_selected,
                                    )
                                if agent_cfg is not None and prev_topk is not None:
                                    setattr(agent_cfg, "pdm_val_score_topk", prev_topk)

                            if record_mode_oracle_metrics and isinstance(
                                oracle_metrics, dict
                            ):
                                oracle_written = False
                                oracle_keys = {
                                    "selected_mode_oracle_rank",
                                    "selected_mode_oracle_percentile",
                                    "selected_mode_oracle_score",
                                    "best_mode_oracle_score",
                                }
                                for key in oracle_keys:
                                    value = oracle_metrics.get(key)
                                    if torch.is_tensor(value):
                                        score_row[f"hc_{key}"] = float(
                                            value.detach().float().mean().item()
                                        )
                                        oracle_written = True
                                    elif isinstance(value, (int, float, bool)):
                                        score_row[f"hc_{key}"] = float(value)
                                        oracle_written = True
                                if not oracle_written and extra_eval_warning_budget > 0:
                                    logger.warning(
                                        "record_mode_oracle_metrics=True but no oracle metrics were produced for token %s.",
                                        token,
                                    )
                                    extra_eval_warning_budget -= 1

                        if record_attention_loss_metrics:
                            attn_loss_fn = getattr(
                                agent, "_compute_mode_bev_attention_aux_loss", None
                            )
                            if callable(attn_loss_fn):
                                attn_aux = attn_loss_fn(eval_predictions)
                                if torch.is_tensor(attn_aux):
                                    score_row["hc_attn_aux"] = float(
                                        attn_aux.detach().float().mean().item()
                                    )
                                elif extra_eval_warning_budget > 0:
                                    logger.warning(
                                        "record_attention_loss_metrics=True but attention aux loss was unavailable for token %s.",
                                        token,
                                    )
                                    extra_eval_warning_budget -= 1
                            elif extra_eval_warning_budget > 0:
                                logger.warning(
                                    "record_attention_loss_metrics=True but agent has no _compute_mode_bev_attention_aux_loss for token %s.",
                                    token,
                                )
                                extra_eval_warning_budget -= 1
                except Exception:
                    if extra_eval_warning_budget > 0:
                        logger.warning(
                            "Failed to collect extra eval metrics for token %s",
                            token,
                            exc_info=True,
                        )
                        extra_eval_warning_budget -= 1
                    else:
                        logger.debug(
                            "Failed to collect extra eval metrics for token %s",
                            token,
                            exc_info=True,
                        )

            if record_hardcase_metrics:
                debug_getter = getattr(agent, "get_last_inference_debug", None)
                if callable(debug_getter):
                    try:
                        debug_metrics = debug_getter()
                        if isinstance(debug_metrics, dict):
                            for k, v in debug_metrics.items():
                                if isinstance(v, (int, float, bool)):
                                    score_row[k] = float(v)
                    except Exception:
                        logger.debug("Failed to collect hardcase debug metrics for token %s", token)
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    token_subset_path = cfg.get("token_subset_path", None)
    token_subset = None
    if token_subset_path:
        token_subset_key = str(cfg.get("token_subset_key", "tokens"))
        token_subset = _load_token_subset(str(token_subset_path), key=token_subset_key)
        logger.info(
            "Loaded token subset from %s with %d tokens.",
            token_subset_path,
            len(token_subset),
        )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    prefer_training_cache = bool(cfg.get("prefer_training_cache", False))
    training_cache_path = cfg.get("training_cache_path", None)

    tokens_per_log: Dict[str, List[str]]
    if prefer_training_cache and training_cache_path and Path(str(training_cache_path)).is_dir():
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        tokens_per_log = _load_cached_tokens_per_log(
            cache_root=Path(str(training_cache_path)),
            log_names=list(scene_filter.log_names) if scene_filter.log_names is not None else None,
            tokens=token_subset if token_subset is not None else scene_filter.tokens,
            max_scenes=scene_filter.max_scenes,
        )
        logger.info(
            "Using training cache to enumerate tokens from %s across %d logs.",
            training_cache_path,
            len(tokens_per_log),
        )
    else:
        # Extract scenes based on scene-loader to know which tokens to distribute across workers
        # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        if token_subset is not None:
            scene_filter.tokens = token_subset
        scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        tokens_per_log = scene_loader.get_tokens_list_per_log()

    metric_tokens = set(metric_cache_loader.tokens)
    filtered_tokens_per_log = {
        log_file: [token for token in tokens if token in metric_tokens]
        for log_file, tokens in tokens_per_log.items()
    }
    filtered_tokens_per_log = {
        log_file: tokens for log_file, tokens in filtered_tokens_per_log.items() if tokens
    }
    all_filtered_tokens = [token for tokens in filtered_tokens_per_log.values() for token in tokens]
    all_source_tokens = [token for tokens in tokens_per_log.values() for token in tokens]
    tokens_to_evaluate = list(all_filtered_tokens)
    num_missing_metric_cache_tokens = len(set(all_source_tokens) - metric_tokens)
    num_unused_metric_cache_tokens = len(metric_tokens - set(all_source_tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in filtered_tokens_per_log.items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
