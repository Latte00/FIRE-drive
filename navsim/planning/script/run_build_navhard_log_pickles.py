from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


@dataclass
class BuildStats:
    input_files: int = 0
    loaded_files: int = 0
    skipped_files: int = 0
    total_frames_seen: int = 0
    unique_frames_kept: int = 0
    output_logs: int = 0


def _normalize_log_name(log_name: Any) -> Optional[str]:
    if not isinstance(log_name, str):
        return None
    name = Path(log_name).name
    if name.endswith(".pkl"):
        name = name[:-4]
    if name.endswith(".db"):
        name = name[:-3]
    if not name:
        return None
    return name


def _unwrap_scene_list(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return obj
        return None
    if isinstance(obj, dict):
        scene_list = obj.get("scene_dict_list")
        if isinstance(scene_list, list) and all(isinstance(x, dict) for x in scene_list):
            return scene_list
        if len(obj) == 1:
            only_val = next(iter(obj.values()))
            if isinstance(only_val, list) and all(isinstance(x, dict) for x in only_val):
                return only_val
    return None


def _iter_input_pickles(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.pkl"))


def build_log_pickles(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    mapping_csv: Optional[Path] = None,
) -> BuildStats:
    stats = BuildStats()
    output_dir.mkdir(parents=True, exist_ok=True)

    # normalized_log_name -> token -> frame_dict
    log_token_to_frame: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    # token -> (raw_log_name, normalized_log_name, timestamp, source_file)
    token_rows: List[Tuple[str, str, str, int, str]] = []

    files = list(_iter_input_pickles(input_dir))
    stats.input_files = len(files)

    for pkl_path in tqdm(files, desc="Loading scene snippets"):
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            scene_list = _unwrap_scene_list(obj)
            if not scene_list:
                stats.skipped_files += 1
                continue
            stats.loaded_files += 1

            for frame in scene_list:
                stats.total_frames_seen += 1
                token = frame.get("token")
                raw_log_name = frame.get("log_name")
                norm_log_name = _normalize_log_name(raw_log_name)
                timestamp = frame.get("timestamp")
                if not isinstance(token, str) or norm_log_name is None:
                    continue
                if not isinstance(timestamp, (int, float)):
                    timestamp = -1
                timestamp = int(timestamp)

                if token not in log_token_to_frame[norm_log_name]:
                    # Keep normalized log_name in frame payload for downstream consistency.
                    frame["log_name"] = norm_log_name
                    log_token_to_frame[norm_log_name][token] = frame
                    stats.unique_frames_kept += 1
                    token_rows.append(
                        (
                            token,
                            str(raw_log_name),
                            norm_log_name,
                            timestamp,
                            pkl_path.name,
                        )
                    )
        except Exception:
            stats.skipped_files += 1
            continue

    for log_name, token_to_frame in tqdm(log_token_to_frame.items(), desc="Writing merged logs"):
        out_path = output_dir / f"{log_name}.pkl"
        if out_path.exists() and not overwrite:
            continue

        frames = list(token_to_frame.values())
        frames.sort(key=lambda x: int(x.get("timestamp", -1)))
        with open(out_path, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        stats.output_logs += 1

    if mapping_csv is not None:
        mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "token",
                    "raw_log_name",
                    "normalized_log_name",
                    "timestamp",
                    "source_file",
                ]
            )
            writer.writerows(token_rows)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build log-level NAVSIM pickle files from navhard two-stage scene snippets "
            "(e.g., openscene_meta_datas with per-file 5-frame lists)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing snippet .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write merged log-level .pkl files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged output files.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Optional CSV path to export token->log_name mapping.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input dir does not exist: {args.input_dir}")

    stats = build_log_pickles(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        mapping_csv=args.mapping_csv,
    )
    print("Build finished:")
    print(f"  input_files:       {stats.input_files}")
    print(f"  loaded_files:      {stats.loaded_files}")
    print(f"  skipped_files:     {stats.skipped_files}")
    print(f"  total_frames_seen: {stats.total_frames_seen}")
    print(f"  unique_frames:     {stats.unique_frames_kept}")
    print(f"  output_logs:       {stats.output_logs}")
    print(f"  output_dir:        {args.output_dir}")
    if args.mapping_csv is not None:
        print(f"  mapping_csv:       {args.mapping_csv}")


if __name__ == "__main__":
    main()
