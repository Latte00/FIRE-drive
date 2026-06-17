from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


# Allow direct script execution (`python navsim/planning/script/...py`) to
# unpickle metric_cache objects that reference the `navsim` package.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build hard + nearby-easy training subsets from run_pdm_score CSV, "
            "existing hardset json, and metric cache timing metadata."
        )
    )
    parser.add_argument("--score-csv", type=Path, required=True, help="Input run_pdm_score csv.")
    parser.add_argument("--hardset-json", type=Path, required=True, help="Existing hardset json.")
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        required=True,
        help="Metric cache root used to recover log/time for each token.",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Output paired-set json.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional pairwise csv with hard/easy assignments.",
    )
    parser.add_argument(
        "--time-window-s",
        type=float,
        default=10.0,
        help="Nearby easy search window in seconds within the same log.",
    )
    parser.add_argument(
        "--max-easy-per-hard",
        type=int,
        default=4,
        help="Maximum nearby easy samples to attach to each hard token.",
    )
    parser.add_argument(
        "--max-easy-reuse",
        type=int,
        default=2,
        help="Maximum reuse count for the same easy token across hard tokens.",
    )
    parser.add_argument(
        "--easy-score-min",
        type=float,
        default=0.90,
        help="Minimum score for easy candidates.",
    )
    parser.add_argument(
        "--easy-u-max",
        type=float,
        default=0.35,
        help="Maximum hc_u_adapter_variance for easy candidates. Ignored if column is missing.",
    )
    parser.add_argument(
        "--easy-bev-ce-max",
        type=float,
        default=None,
        help="Maximum hc_bev_ce for easy candidates. Ignored if column is missing or unset.",
    )
    parser.add_argument(
        "--global-fill-count",
        type=int,
        default=0,
        help="If nearby easy samples are insufficient, fill up to this many with global easy samples.",
    )
    parser.add_argument(
        "--contiguous-segment-enable",
        action="store_true",
        help="Select nearby easy samples as a contiguous temporal segment around each hard token.",
    )
    parser.add_argument(
        "--full-segment-enable",
        action="store_true",
        help=(
            "Take the full valid non-hard segment within the same-log time window for each hard token. "
            "When enabled, easy-score/u/bev filters, max-easy-per-hard, max-easy-reuse, and global fill "
            "are ignored for segment selection."
        ),
    )
    parser.add_argument(
        "--global-fill-random",
        action="store_true",
        help="Randomly sample global-fill easy scenes instead of deterministically taking the top sorted ones.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used when global-fill random sampling is enabled.",
    )
    return parser.parse_args()


def _load_hard_tokens(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"hardset json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("hard_tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"hard_tokens missing in {path}")
    return [str(token) for token in tokens]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _optional_numeric(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    return _to_numeric(df[column])


def _load_metric_cache_paths(metric_cache_path: Path) -> Dict[str, str]:
    metric_cache_paths: Dict[str, str] = {}

    metadata_dir = metric_cache_path / "metadata"
    if metadata_dir.exists():
        metadata_files = sorted(metadata_dir.glob("*.csv"))
        for metadata_file in metadata_files:
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if not row:
                            continue
                        cache_path_str = str(row[0]).strip()
                        if not cache_path_str:
                            continue
                        token = Path(cache_path_str).parent.name
                        if token:
                            metric_cache_paths[token] = cache_path_str
            except Exception:
                continue

    # Fallback: build token index by scanning the cache tree directly.
    if not metric_cache_paths:
        for cache_file in metric_cache_path.rglob("metric_cache.pkl"):
            token = cache_file.parent.name
            if token:
                metric_cache_paths[token] = str(cache_file)

    if not metric_cache_paths:
        raise ValueError(f"no metric cache paths found under: {metric_cache_path}")
    return metric_cache_paths


def _load_metric_cache_index(metric_cache_path: Path, tokens: Sequence[str]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    metric_cache_paths = _load_metric_cache_paths(metric_cache_path)
    rows: List[Dict[str, Any]] = []
    token_set: Set[str] = set(str(token) for token in tokens)
    matched_tokens = sorted(token_set & set(metric_cache_paths.keys()))
    load_error_count = 0
    sample_load_errors: List[str] = []

    for token in matched_tokens:
        cache_path = Path(metric_cache_paths[token])
        log_name = cache_path.parents[2].name if len(cache_path.parents) >= 3 else ""
        scenario_type = cache_path.parents[1].name if len(cache_path.parents) >= 2 else ""
        time_s = np.nan
        try:
            with lzma.open(cache_path, "rb") as f:
                metric_cache = pickle.load(f)
            ego_state = getattr(metric_cache, "ego_state", None)
            time_point = getattr(ego_state, "time_point", None)
            time_s_attr = getattr(time_point, "time_s", None)
            if time_s_attr is not None:
                time_s = float(time_s_attr)
            else:
                time_us_attr = getattr(time_point, "time_us", None)
                if time_us_attr is not None:
                    time_s = float(time_us_attr) / 1e6
        except Exception as exc:
            load_error_count += 1
            if len(sample_load_errors) < 10:
                sample_load_errors.append(f"{token}: {type(exc).__name__}: {exc}")
            time_s = np.nan

        rows.append(
            {
                "token": token,
                "metric_log_name": log_name,
                "metric_scenario_type": scenario_type,
                "metric_time_s": time_s,
                "metric_cache_file": str(cache_path),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "token",
            "metric_log_name",
            "metric_scenario_type",
            "metric_time_s",
            "metric_cache_file",
        ],
    )
    debug = {
        "metric_load_error_count": int(load_error_count),
        "sample_metric_load_errors": sample_load_errors,
    }
    return frame, debug


def _count_metric_token_hits(metric_cache_path: Path, tokens: Sequence[str]) -> Dict[str, Any]:
    metric_cache_paths = _load_metric_cache_paths(metric_cache_path)
    token_set: Set[str] = set(str(token) for token in tokens)
    matched_tokens = sorted(token_set & set(metric_cache_paths.keys()))
    missing_tokens = sorted(token_set - set(metric_cache_paths.keys()))
    return {
        "num_csv_tokens": int(len(token_set)),
        "num_metric_index_tokens": int(len(metric_cache_paths)),
        "num_metric_token_hits": int(len(matched_tokens)),
        "sample_missing_metric_tokens": missing_tokens[:10],
    }


def _sort_easy_candidates(df: pd.DataFrame, delta_col: Optional[str] = None) -> pd.DataFrame:
    sort_cols: List[str] = []
    ascending: List[bool] = []
    if delta_col is not None and delta_col in df.columns:
        sort_cols.append(delta_col)
        ascending.append(True)
    if "score" in df.columns:
        sort_cols.append("score")
        ascending.append(False)
    if "hc_u_adapter_variance" in df.columns:
        sort_cols.append("hc_u_adapter_variance")
        ascending.append(True)
    if "hc_bev_ce" in df.columns:
        sort_cols.append("hc_bev_ce")
        ascending.append(True)
    if "metric_time_s" in df.columns:
        sort_cols.append("metric_time_s")
        ascending.append(True)
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def _select_tokens(
    candidates: pd.DataFrame,
    num_to_select: int,
    reuse_counter: Counter,
    max_reuse: int,
) -> List[str]:
    selected: List[str] = []
    for token in candidates["token"].astype(str).tolist():
        if reuse_counter[token] >= max_reuse:
            continue
        selected.append(token)
        reuse_counter[token] += 1
        if len(selected) >= num_to_select:
            break
    return selected


def _select_tokens_random(
    candidates: pd.DataFrame,
    num_to_select: int,
    reuse_counter: Counter,
    max_reuse: int,
    rng: np.random.Generator,
) -> List[str]:
    tokens = candidates["token"].astype(str).tolist()
    if not tokens or num_to_select <= 0:
        return []
    order = rng.permutation(len(tokens))
    selected: List[str] = []
    for idx in order.tolist():
        token = tokens[idx]
        if reuse_counter[token] >= max_reuse:
            continue
        selected.append(token)
        reuse_counter[token] += 1
        if len(selected) >= num_to_select:
            break
    return selected


def _select_contiguous_tokens(
    candidates: pd.DataFrame,
    center_time_s: float,
    num_to_select: int,
    reuse_counter: Counter,
    max_reuse: int,
) -> List[str]:
    if num_to_select <= 0 or len(candidates) <= 0:
        return []
    if "metric_time_s" not in candidates.columns:
        return _select_tokens(candidates, num_to_select, reuse_counter, max_reuse)

    timed = candidates.copy()
    timed["metric_time_s"] = pd.to_numeric(timed["metric_time_s"], errors="coerce")
    timed = timed[timed["metric_time_s"].notna()].sort_values(
        "metric_time_s", ascending=True, kind="mergesort"
    )
    if len(timed) == 0:
        return _select_tokens(candidates, num_to_select, reuse_counter, max_reuse)

    times = timed["metric_time_s"].to_numpy(dtype=np.float64)
    insert_idx = int(np.searchsorted(times, float(center_time_s)))
    left = insert_idx - 1
    right = insert_idx
    visit_order: List[int] = []

    while left >= 0 or right < len(times):
        left_delta = (
            abs(times[left] - float(center_time_s)) if left >= 0 else np.inf
        )
        right_delta = (
            abs(times[right] - float(center_time_s)) if right < len(times) else np.inf
        )
        if left_delta <= right_delta:
            visit_order.append(left)
            left -= 1
        else:
            visit_order.append(right)
            right += 1

    selected: List[str] = []
    tokens = timed["token"].astype(str).tolist()
    for idx in visit_order:
        token = tokens[idx]
        if reuse_counter[token] >= max_reuse:
            continue
        selected.append(token)
        reuse_counter[token] += 1
        if len(selected) >= num_to_select:
            break
    return selected


def _select_full_segment_tokens(
    candidates: pd.DataFrame,
    center_time_s: Optional[float],
) -> List[str]:
    if len(candidates) <= 0:
        return []

    segment_df = candidates.copy()
    if (
        center_time_s is not None
        and "metric_time_s" in segment_df.columns
    ):
        segment_df["metric_time_s"] = pd.to_numeric(
            segment_df["metric_time_s"], errors="coerce"
        )
        segment_df = segment_df.sort_values(
            "metric_time_s", ascending=True, kind="mergesort"
        )
    else:
        segment_df = _sort_easy_candidates(segment_df)

    return segment_df["token"].astype(str).tolist()


def main() -> None:
    args = _parse_args()
    if not args.score_csv.exists():
        raise FileNotFoundError(f"score csv not found: {args.score_csv}")

    hard_tokens = _load_hard_tokens(args.hardset_json)
    hard_token_set = set(hard_tokens)

    df = pd.read_csv(args.score_csv)
    if "token" not in df.columns:
        raise ValueError("CSV must contain token column")
    df = df[df["token"].astype(str) != "average"].copy()
    df["token"] = df["token"].astype(str)

    metric_hit_stats = _count_metric_token_hits(args.metric_cache_path, df["token"].tolist())
    metric_meta, metric_debug = _load_metric_cache_index(args.metric_cache_path, df["token"].tolist())
    merged = df.merge(metric_meta, on="token", how="left")

    valid_series = (
        (_to_numeric(merged["valid"]).fillna(0.0) > 0.5)
        if "valid" in merged.columns
        else pd.Series(np.ones(len(merged), dtype=bool), index=merged.index)
    )
    score_series = _optional_numeric(merged, "score")
    u_series = _optional_numeric(merged, "hc_u_adapter_variance")
    bev_series = _optional_numeric(merged, "hc_bev_ce")

    hard_df = merged[merged["token"].isin(hard_token_set)].copy()
    if len(hard_df) == 0:
        raise ValueError("No hard tokens from hardset were found in score csv.")

    non_hard_mask = ~merged["token"].isin(hard_token_set)
    easy_mask = non_hard_mask & valid_series
    score_pass_mask = (
        (score_series >= float(args.easy_score_min))
        if score_series is not None
        else pd.Series(np.ones(len(merged), dtype=bool), index=merged.index)
    )
    u_pass_mask = (
        (u_series <= float(args.easy_u_max))
        if (u_series is not None and args.easy_u_max is not None)
        else pd.Series(np.ones(len(merged), dtype=bool), index=merged.index)
    )
    bev_pass_mask = (
        (bev_series <= float(args.easy_bev_ce_max))
        if (bev_series is not None and args.easy_bev_ce_max is not None)
        else pd.Series(np.ones(len(merged), dtype=bool), index=merged.index)
    )
    easy_mask &= score_pass_mask & u_pass_mask & bev_pass_mask

    easy_df = merged[easy_mask].copy()
    global_easy_df = _sort_easy_candidates(easy_df.copy())
    valid_non_hard_df = merged[non_hard_mask & valid_series].copy()
    hard_df = hard_df.sort_values(["metric_log_name", "metric_time_s"], kind="mergesort")

    reuse_counter: Counter = Counter()
    pair_rows: List[Dict[str, Any]] = []
    hard_to_easy_map: Dict[str, List[str]] = {}

    max_easy_per_hard = max(int(args.max_easy_per_hard), 0)
    max_easy_reuse = max(int(args.max_easy_reuse), 1)
    global_fill_count = max(int(args.global_fill_count), 0)
    time_window_s = max(float(args.time_window_s), 0.0)
    rng = np.random.default_rng(int(args.random_seed))

    for _, hard_row in hard_df.iterrows():
        hard_token = str(hard_row["token"])
        hard_log_name = hard_row.get("metric_log_name", "")
        hard_time_s = hard_row.get("metric_time_s", np.nan)
        selected_tokens: List[str] = []

        if bool(args.full_segment_enable):
            segment_df = valid_non_hard_df.iloc[0:0].copy()
            if isinstance(hard_log_name, str) and hard_log_name:
                same_log_df = valid_non_hard_df[
                    valid_non_hard_df["metric_log_name"] == hard_log_name
                ].copy()
                segment_df = same_log_df
                if pd.notna(hard_time_s) and "metric_time_s" in segment_df.columns:
                    segment_df["delta_time_s"] = (
                        segment_df["metric_time_s"] - float(hard_time_s)
                    ).abs()
                    segment_df = segment_df[
                        segment_df["delta_time_s"] <= time_window_s
                    ].copy()
            selected_tokens = _select_full_segment_tokens(
                candidates=segment_df,
                center_time_s=float(hard_time_s) if pd.notna(hard_time_s) else None,
            )
            for easy_token in selected_tokens:
                reuse_counter[easy_token] += 1
        else:
            nearby_df = easy_df.iloc[0:0].copy()
            if isinstance(hard_log_name, str) and hard_log_name:
                same_log_df = easy_df[easy_df["metric_log_name"] == hard_log_name].copy()
                nearby_df = same_log_df
                if pd.notna(hard_time_s) and "metric_time_s" in nearby_df.columns:
                    nearby_df["delta_time_s"] = (nearby_df["metric_time_s"] - float(hard_time_s)).abs()
                    nearby_df = nearby_df[nearby_df["delta_time_s"] <= time_window_s].copy()
                    if len(nearby_df) > 0:
                        nearby_df = _sort_easy_candidates(nearby_df, delta_col="delta_time_s")
                    else:
                        # Fall back to same-log matching when timing metadata is unavailable
                        # or no same-log easy sample falls inside the requested window.
                        nearby_df = _sort_easy_candidates(same_log_df)

            if max_easy_per_hard > 0 and len(nearby_df) > 0:
                if bool(args.contiguous_segment_enable) and pd.notna(hard_time_s):
                    selected_tokens.extend(
                        _select_contiguous_tokens(
                            candidates=nearby_df,
                            center_time_s=float(hard_time_s),
                            num_to_select=max_easy_per_hard,
                            reuse_counter=reuse_counter,
                            max_reuse=max_easy_reuse,
                        )
                    )
                else:
                    selected_tokens.extend(
                        _select_tokens(
                            candidates=nearby_df,
                            num_to_select=max_easy_per_hard,
                            reuse_counter=reuse_counter,
                            max_reuse=max_easy_reuse,
                        )
                    )

            fill_budget = min(global_fill_count, max(max_easy_per_hard - len(selected_tokens), 0))
            if fill_budget > 0 and len(global_easy_df) > 0:
                global_candidates = global_easy_df[~global_easy_df["token"].isin(selected_tokens)].copy()
                if bool(args.global_fill_random):
                    selected_tokens.extend(
                        _select_tokens_random(
                            candidates=global_candidates,
                            num_to_select=fill_budget,
                            reuse_counter=reuse_counter,
                            max_reuse=max_easy_reuse,
                            rng=rng,
                        )
                    )
                else:
                    selected_tokens.extend(
                        _select_tokens(
                            candidates=global_candidates,
                            num_to_select=fill_budget,
                            reuse_counter=reuse_counter,
                            max_reuse=max_easy_reuse,
                        )
                    )

        hard_to_easy_map[hard_token] = selected_tokens
        for order_idx, easy_token in enumerate(selected_tokens):
            easy_row = merged[merged["token"] == easy_token].iloc[0]
            source = "nearby"
            delta_time_s = np.nan
            if bool(args.full_segment_enable):
                source = "full_segment"
            if (
                str(easy_row.get("metric_log_name", "")) != str(hard_log_name)
                or pd.isna(easy_row.get("metric_time_s", np.nan))
                or pd.isna(hard_time_s)
            ):
                source = "same_log_no_time" if str(easy_row.get("metric_log_name", "")) == str(hard_log_name) else "global_fill"
            else:
                delta_time_s = abs(float(easy_row["metric_time_s"]) - float(hard_time_s))
                if delta_time_s > time_window_s:
                    source = "same_log_no_time" if str(easy_row.get("metric_log_name", "")) == str(hard_log_name) else "global_fill"
            if source == "global_fill" and bool(args.global_fill_random):
                source = "global_fill_random"

            pair_rows.append(
                {
                    "hard_token": hard_token,
                    "hard_log_name": hard_log_name,
                    "hard_time_s": hard_time_s,
                    "hard_score": hard_row.get("score", np.nan),
                    "hard_hc_u_adapter_variance": hard_row.get("hc_u_adapter_variance", np.nan),
                    "hard_hc_bev_ce": hard_row.get("hc_bev_ce", np.nan),
                    "easy_token": easy_token,
                    "easy_log_name": easy_row.get("metric_log_name", ""),
                    "easy_time_s": easy_row.get("metric_time_s", np.nan),
                    "easy_score": easy_row.get("score", np.nan),
                    "easy_hc_u_adapter_variance": easy_row.get("hc_u_adapter_variance", np.nan),
                    "easy_hc_bev_ce": easy_row.get("hc_bev_ce", np.nan),
                    "delta_time_s": delta_time_s,
                    "pair_source": source,
                    "pair_rank": int(order_idx),
                }
            )

    paired_easy_tokens = sorted({row["easy_token"] for row in pair_rows})
    mixed_tokens = list(dict.fromkeys(hard_tokens + paired_easy_tokens))
    hard_without_easy = [token for token, easy_list in hard_to_easy_map.items() if len(easy_list) == 0]

    output = {
        "source_csv": str(args.score_csv),
        "hardset_json": str(args.hardset_json),
        "metric_cache_path": str(args.metric_cache_path),
        "time_window_s": time_window_s,
        "max_easy_per_hard": max_easy_per_hard,
        "max_easy_reuse": max_easy_reuse,
        "easy_score_min": float(args.easy_score_min),
        "easy_u_max": None if args.easy_u_max is None else float(args.easy_u_max),
        "easy_bev_ce_max": (
            None if args.easy_bev_ce_max is None else float(args.easy_bev_ce_max)
        ),
        "global_fill_count": global_fill_count,
        "contiguous_segment_enable": bool(args.contiguous_segment_enable),
        "full_segment_enable": bool(args.full_segment_enable),
        "global_fill_random": bool(args.global_fill_random),
        "random_seed": int(args.random_seed),
        "num_csv_rows": int(len(merged)),
        "num_csv_tokens": int(metric_hit_stats["num_csv_tokens"]),
        "num_metric_index_tokens": int(metric_hit_stats["num_metric_index_tokens"]),
        "num_metric_token_hits": int(metric_hit_stats["num_metric_token_hits"]),
        "sample_missing_metric_tokens": metric_hit_stats["sample_missing_metric_tokens"],
        "num_valid_rows": int(valid_series.sum()),
        "num_metric_rows": int(len(metric_meta)),
        "num_rows_with_metric_log": int(metric_meta["metric_log_name"].astype(str).ne("").sum())
        if "metric_log_name" in metric_meta.columns
        else 0,
        "num_rows_with_metric_time": int(pd.to_numeric(metric_meta["metric_time_s"], errors="coerce").notna().sum())
        if "metric_time_s" in metric_meta.columns
        else 0,
        "metric_load_error_count": int(metric_debug["metric_load_error_count"]),
        "sample_metric_load_errors": metric_debug["sample_metric_load_errors"],
        "num_hard": int(len(hard_tokens)),
        "num_non_hard_rows": int(non_hard_mask.sum()),
        "num_easy_valid_rows": int((non_hard_mask & valid_series).sum()),
        "num_easy_score_pass": int((non_hard_mask & valid_series & score_pass_mask).sum()),
        "num_easy_u_pass": int((non_hard_mask & valid_series & score_pass_mask & u_pass_mask).sum()),
        "num_easy_bev_pass": int(
            (non_hard_mask & valid_series & score_pass_mask & u_pass_mask & bev_pass_mask).sum()
        ),
        "num_easy_candidates": int(len(easy_df)),
        "num_pairs": int(len(pair_rows)),
        "num_paired_easy_unique": int(len(paired_easy_tokens)),
        "num_hard_without_easy": int(len(hard_without_easy)),
        "avg_easy_per_hard": float(len(pair_rows) / max(len(hard_tokens), 1)),
        "hard_tokens": hard_tokens,
        "paired_easy_tokens": paired_easy_tokens,
        "mixed_tokens": mixed_tokens,
        "hard_without_easy_tokens": hard_without_easy,
        "hard_to_easy_map": hard_to_easy_map,
        "easy_reuse_counter": dict(sorted(reuse_counter.items())),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_csv is not None:
        pair_df = pd.DataFrame(pair_rows)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pair_df.to_csv(args.out_csv, index=False)

    print(
        f"[Hard+Easy] hard={len(hard_tokens)} easy_candidates={len(easy_df)} "
        f"pairs={len(pair_rows)} unique_easy={len(paired_easy_tokens)} "
        f"hard_without_easy={len(hard_without_easy)}"
    )
    print(
        f"[Hard+Easy] csv_tokens={metric_hit_stats['num_csv_tokens']} "
        f"metric_index_tokens={metric_hit_stats['num_metric_index_tokens']} "
        f"metric_hits={metric_hit_stats['num_metric_token_hits']}"
    )
    print(
        f"[Hard+Easy] metric_load_errors={int(metric_debug['metric_load_error_count'])}"
    )
    print(
        f"[Hard+Easy] valid={int(valid_series.sum())} non_hard={int(non_hard_mask.sum())} "
        f"score_pass={int((non_hard_mask & valid_series & score_pass_mask).sum())} "
        f"u_pass={int((non_hard_mask & valid_series & score_pass_mask & u_pass_mask).sum())} "
        f"bev_pass={int((non_hard_mask & valid_series & score_pass_mask & u_pass_mask & bev_pass_mask).sum())}"
    )
    print(f"[Hard+Easy] json: {args.out_json}")
    if args.out_csv is not None:
        print(f"[Hard+Easy] csv : {args.out_csv}")


if __name__ == "__main__":
    main()
