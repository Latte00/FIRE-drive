#!/usr/bin/env python3
"""
Export a dedicated train_test_split config from a training Hydra config.

This is useful to reproduce monitor evaluation logs in run_pdm_score without
editing existing splits such as navtest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got: {type(data)}")
    return data


def _ensure_scene_filter_defaults(scene_filter_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "num_history_frames": scene_filter_cfg.get("num_history_frames", 4),
        "num_future_frames": scene_filter_cfg.get("num_future_frames", 10),
        "frame_interval": scene_filter_cfg.get("frame_interval", 1),
        "has_route": scene_filter_cfg.get("has_route", True),
        "max_scenes": scene_filter_cfg.get("max_scenes", None),
    }
    return out


def _write_yaml(path: Path, payload: Dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Use --force to overwrite this generated file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a new train_test_split from a training Hydra config's test_logs."
        )
    )
    parser.add_argument(
        "--train-hydra-config",
        type=str,
        required=True,
        help="Path to training experiment hydra config.yaml",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        help="Name of generated split (e.g., monitor_eval_20260309)",
    )
    parser.add_argument(
        "--log-key",
        type=str,
        default="test_logs",
        choices=["train_logs", "val_logs", "test_logs"],
        help="Which log list in hydra config to export.",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default=None,
        help=(
            "Optional override for generated split data_split "
            "(e.g., test/trainval)."
        ),
    )
    parser.add_argument(
        "--output-common-dir",
        type=str,
        default="navsim/planning/script/config/common",
        help="Root directory that contains train_test_split/ and scene_filter/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.train_hydra_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training hydra config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    train_test_split_cfg = cfg.get("train_test_split")
    if not isinstance(train_test_split_cfg, dict):
        raise ValueError("Missing train_test_split in training hydra config.")

    scene_filter_cfg = train_test_split_cfg.get("scene_filter")
    if not isinstance(scene_filter_cfg, dict):
        raise ValueError("Missing train_test_split.scene_filter in hydra config.")

    selected_logs = cfg.get(args.log_key)
    if not isinstance(selected_logs, list) or len(selected_logs) == 0:
        raise ValueError(f"No {args.log_key} found in hydra config.")
    selected_logs = [str(x) for x in selected_logs]

    data_split = args.data_split or train_test_split_cfg.get("data_split", "trainval")
    split_name = args.split_name.strip()
    if not split_name:
        raise ValueError("--split-name must be a non-empty string.")

    common_root = Path(args.output_common_dir).expanduser().resolve()
    split_dir = common_root / "train_test_split"
    scene_filter_dir = split_dir / "scene_filter"

    scene_filter_payload = _ensure_scene_filter_defaults(scene_filter_cfg)
    scene_filter_payload["log_names"] = selected_logs
    split_payload = {"defaults": [{"scene_filter": split_name}], "data_split": data_split}

    scene_filter_path = scene_filter_dir / f"{split_name}.yaml"
    split_path = split_dir / f"{split_name}.yaml"

    _write_yaml(scene_filter_path, scene_filter_payload, args.force)
    _write_yaml(split_path, split_payload, args.force)

    print("Generated split successfully.")
    print(f"  split_name: {split_name}")
    print(f"  log_key: {args.log_key}")
    print(f"  data_split: {data_split}")
    print(f"  num_logs: {len(selected_logs)}")
    print(f"  scene_filter_yaml: {scene_filter_path}")
    print(f"  split_yaml: {split_path}")
    print("")
    print("Use it in run_pdm_score:")
    print(f"  train_test_split={split_name}")


if __name__ == "__main__":
    main()
