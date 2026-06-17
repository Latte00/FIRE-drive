from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    for key in list(data_dict.keys()):
        # Exact PDM scene helper objects are useful for offline cache analysis,
        # but PyTorch's default collate cannot batch EgoState/PDM map objects.
        if key.startswith("pdm_scene_"):
            data_dict.pop(key)
    for key, value in data_dict.items():
        if torch.is_tensor(value) and value.is_cuda:
            data_dict[key] = value.cpu()
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
        scorer_cache_path: Optional[str] = None,
        scorer_cache_filename: str = "scorer_cache.gz",
        require_scorer_cache: bool = False,
        validate_cache: bool = False,
        validation_max_attempts: int = 16,
        validation_max_trajectory_translation: float = 120.0,
        validation_max_trajectory_step: float = 60.0,
        validation_max_status_abs: float = 100.0,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        :param tokens: optional list of tokens to consider, defaults to None
        :param scorer_cache_path: optional path to scorer cache directory
        :param scorer_cache_filename: filename for scorer cache entries
        :param require_scorer_cache: if true, keep only tokens that exist in scorer cache
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._validate_cache = bool(validate_cache)
        self._validation_max_attempts = max(1, int(validation_max_attempts))
        self._validation_max_trajectory_translation = float(validation_max_trajectory_translation)
        self._validation_max_trajectory_step = float(validation_max_trajectory_step)
        self._validation_max_status_abs = float(validation_max_status_abs)
        self._invalid_cache_count = 0
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self._scorer_cache_paths: Dict[str, Path] = {}
        if scorer_cache_path:
            self._scorer_cache_paths = self._load_scorer_caches(
                cache_path=Path(scorer_cache_path),
                cache_filename=scorer_cache_filename,
                log_names=self.log_names,
            )
            if require_scorer_cache:
                self._valid_cache_paths = {
                    token: path
                    for token, path in self._valid_cache_paths.items()
                    if token in self._scorer_cache_paths
                }
        if tokens is not None:
            token_set = set(tokens)
            self._valid_cache_paths = {
                token: path
                for token, path in self._valid_cache_paths.items()
                if token in token_set
            }
            if self._scorer_cache_paths:
                self._scorer_cache_paths = {
                    token: path
                    for token, path in self._scorer_cache_paths.items()
                    if token in self._valid_cache_paths
                }
            self.tokens = [token for token in tokens if token in self._valid_cache_paths]
        else:
            self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        if not self._validate_cache:
            return self._load_scene_with_token(self.tokens[idx])

        last_sample = None
        last_reasons: List[str] = []
        num_tokens = len(self.tokens)
        for offset in range(min(self._validation_max_attempts, num_tokens)):
            token = self.tokens[(idx + offset) % num_tokens]
            sample = self._load_scene_with_token(token)
            reasons = self._validate_sample(sample)
            if not reasons:
                return sample
            last_sample = sample
            last_reasons = reasons
            self._invalid_cache_count += 1
            if self._invalid_cache_count <= 20 or self._invalid_cache_count % 100 == 0:
                logger.warning(
                    "Skipping invalid cache sample token=%s reasons=%s invalid_count=%d",
                    token,
                    ",".join(reasons),
                    self._invalid_cache_count,
                )

        if last_sample is None:
            return self._load_scene_with_token(self.tokens[idx])
        logger.warning(
            "No valid replacement found after %d attempts at idx=%d. Returning sanitized fallback.",
            self._validation_max_attempts,
            idx,
        )
        return self._sanitize_sample(last_sample)

    @staticmethod
    def _tensor_is_finite(value: object) -> bool:
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all().item())
        return True

    def _validate_sample(
        self,
        sample: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> List[str]:
        features, targets = sample
        reasons: List[str] = []

        for key in ("status_feature", "ego_history", "lidar_feature"):
            value = features.get(key)
            if value is not None and not self._tensor_is_finite(value):
                reasons.append(f"nonfinite_feature:{key}")

        for key in (
            "trajectory",
            "agent_states",
            "agent_labels",
            "future_agent_boxes",
            "future_agent_boxes_mask",
            "bev_semantic_map",
            "future_bev_semantic_map",
            "feasible_area_mask",
            "feasible_lane_mask",
            "trajectory_candidates",
            "trajectory_candidates_mask",
            "gt_pdm_score",
        ):
            value = targets.get(key)
            if value is not None and torch.is_tensor(value) and value.numel() > 0:
                if not self._tensor_is_finite(value):
                    reasons.append(f"nonfinite_target:{key}")

        score_targets = targets.get("pdm_score_targets")
        if torch.is_tensor(score_targets) and score_targets.numel() > 0:
            scores = score_targets
            cand_mask = targets.get("trajectory_candidates_mask")
            if torch.is_tensor(cand_mask) and cand_mask.numel() == scores.numel():
                active_scores = scores.reshape(-1)[cand_mask.reshape(-1).bool()]
                if active_scores.numel() > 0 and not self._tensor_is_finite(active_scores):
                    reasons.append("nonfinite_target:pdm_score_targets_active")
            elif not self._tensor_is_finite(scores):
                reasons.append("nonfinite_target:pdm_score_targets")

        trajectory = targets.get("trajectory")
        if not torch.is_tensor(trajectory):
            reasons.append("missing_target:trajectory")
        elif trajectory.ndim != 2 or trajectory.shape[-1] < 2 or trajectory.numel() == 0:
            reasons.append(f"bad_shape:trajectory:{tuple(trajectory.shape)}")
        elif torch.isfinite(trajectory[:, :2]).all().item():
            xy = trajectory[:, :2].float()
            if self._validation_max_trajectory_translation > 0:
                max_translation = float(torch.linalg.norm(xy, dim=-1).max().item())
                if max_translation > self._validation_max_trajectory_translation:
                    reasons.append(f"outlier:trajectory_translation:{max_translation:.2f}")
            if self._validation_max_trajectory_step > 0 and xy.shape[0] > 1:
                max_step = float(torch.linalg.norm(xy[1:] - xy[:-1], dim=-1).max().item())
                if max_step > self._validation_max_trajectory_step:
                    reasons.append(f"outlier:trajectory_step:{max_step:.2f}")

        status = features.get("status_feature")
        if torch.is_tensor(status) and status.numel() > 0 and torch.isfinite(status).all().item():
            max_status = float(status.float().abs().max().item())
            if self._validation_max_status_abs > 0 and max_status > self._validation_max_status_abs:
                reasons.append(f"outlier:status_feature:{max_status:.2f}")

        return reasons

    @staticmethod
    def _sanitize_sample(
        sample: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        features, targets = sample
        clean_features: Dict[str, torch.Tensor] = {}
        clean_targets: Dict[str, torch.Tensor] = {}
        for key, value in features.items():
            if torch.is_tensor(value) and value.is_floating_point():
                clean_features[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                clean_features[key] = value
        for key, value in targets.items():
            if torch.is_tensor(value) and value.is_floating_point():
                clean_targets[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                clean_targets[key] = value
        return clean_features, clean_targets

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    @staticmethod
    def _load_scorer_caches(
        cache_path: Path,
        cache_filename: str,
        log_names: Optional[List[Path]] = None,
    ) -> Dict[str, Path]:
        """
        Helper method to load scorer cache paths.
        :param cache_path: directory of scorer cache folder
        :param cache_filename: filename for scorer cache entries
        :param log_names: optional list of log paths to load
        :return: dictionary of tokens and scorer cache paths
        """
        scorer_cache_paths: Dict[str, Path] = {}
        if not cache_path.is_dir():
            return scorer_cache_paths
        if log_names is not None:
            log_paths = [
                Path(log_name)
                for log_name in log_names
                if (cache_path / log_name).is_dir()
            ]
        else:
            log_paths = [log_path for log_path in cache_path.iterdir() if log_path.is_dir()]
        for log_name in log_paths:
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                scorer_path = token_path / cache_filename
                if scorer_path.is_file():
                    scorer_cache_paths[token_path.name] = scorer_path
        return scorer_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        scorer_path = self._scorer_cache_paths.get(token)
        if scorer_path is not None:
            scorer_dict = load_feature_target_from_pickle(scorer_path)
            targets.update(scorer_dict)

        return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)
