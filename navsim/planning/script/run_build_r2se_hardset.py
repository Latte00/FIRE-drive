from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_fraction_list(value: str) -> List[float]:
    rates: List[float] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        rate = float(item)
        if not np.isfinite(rate):
            raise ValueError(f"Invalid target hard rate: {raw_item}")
        rates.append(rate)
    if not rates:
        raise ValueError("At least one target hard rate must be provided.")
    return rates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build hard-case subset from run_pdm_score CSV. Legacy mode uses "
            "FX = (1 - F_plan) + beta_per * F_per + beta_bev * F_bev + beta_ent * F_ent; "
            "regret_risk mode uses rank-based regret and risk signals."
        )
    )
    parser.add_argument("--score-csv", type=Path, required=True, help="Input run_pdm_score csv.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output hardset json path.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output csv for selected hard samples.",
    )
    parser.add_argument(
        "--eps-percent",
        type=float,
        default=1.0,
        help="Top eps%% hardest samples to keep (e.g. 1.0 means top 1%%).",
    )
    parser.add_argument(
        "--hardness-mode",
        type=str,
        choices=("legacy", "regret_risk"),
        default="legacy",
        help="Hardness definition. legacy preserves the old FX formula.",
    )
    parser.add_argument(
        "--plan-col",
        type=str,
        default="score",
        help="Column used as F_plan (larger is better).",
    )
    parser.add_argument(
        "--per-col",
        type=str,
        default="",
        help="Optional column used as F_per before normalization.",
    )
    parser.add_argument(
        "--bev-col",
        type=str,
        default="",
        help="Optional column used as F_bev before normalization.",
    )
    parser.add_argument(
        "--ent-col",
        type=str,
        default="hc_u_adapter_variance",
        help="Column used as F_ent. If missing, falls back to hc_u_adapter_variance_raw.",
    )
    parser.add_argument("--beta-per", type=float, default=0.0, help="Weight of F_per term.")
    parser.add_argument("--beta-bev", type=float, default=0.0, help="Weight of F_bev term.")
    parser.add_argument("--beta-ent", type=float, default=0.25, help="Weight of F_ent term.")
    parser.add_argument(
        "--regret-col",
        type=str,
        default="",
        help=(
            "Optional precomputed regret column. When empty, regret_risk mode first tries "
            "hc_best_mode_oracle_score - hc_selected_mode_oracle_score, then falls back to "
            "selected oracle percentile/rank."
        ),
    )
    parser.add_argument(
        "--attn-col",
        type=str,
        default="hc_attn_aux",
        help="Attention auxiliary loss column used in regret_risk mode.",
    )
    parser.add_argument(
        "--score-var-col",
        type=str,
        default="score_mode_var_generalist",
        help="Score variance column used in regret_risk mode.",
    )
    parser.add_argument(
        "--beta-regret",
        type=float,
        default=1.0,
        help="Weight of regret rank term in regret_risk mode.",
    )
    parser.add_argument(
        "--beta-attn",
        type=float,
        default=0.0,
        help="Weight of attention-loss rank term in regret_risk mode.",
    )
    parser.add_argument(
        "--beta-var",
        type=float,
        default=0.0,
        help="Weight of score-variance rank term in regret_risk mode.",
    )
    parser.add_argument(
        "--beta-plan-rank",
        type=float,
        default=0.0,
        help="Optional weight of low-score plan rank term in regret_risk mode.",
    )
    parser.add_argument(
        "--strict-topk",
        action="store_true",
        help=(
            "Select exactly top ceil(eps%% * N) rows by FX instead of all rows "
            "with FX >= quantile threshold. Useful when many rows tie at the threshold."
        ),
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include rows with valid=false in hardset construction.",
    )
    parser.add_argument(
        "--scan-out-csv",
        type=Path,
        default=None,
        help=(
            "Optional output csv that scans multiple target hard rates using the same FX "
            "definition. Rates are decimals, e.g. 0.01 means top 1%% hardest samples."
        ),
    )
    parser.add_argument(
        "--scan-target-hard-rates",
        type=str,
        default="0.0045,0.01,0.02,0.03,0.05",
        help=(
            "Comma-separated target hard-rate fractions for paper-style quantile scan. "
            "Default matches R2SE-style eps={0.45,1,2,3,5}%%."
        ),
    )
    parser.add_argument(
        "--scan-reference-bool-col",
        type=str,
        default="",
        help=(
            "Optional boolean/0-1 reference hard label column for scan diagnostics. "
            "When provided, precision/recall/F1 are computed against this mask."
        ),
    )
    parser.add_argument(
        "--scan-reference-col",
        type=str,
        default="score",
        help=(
            "Optional numeric column used with --scan-reference-threshold to build a "
            "reference hard mask. Ignored when --scan-reference-bool-col is set."
        ),
    )
    parser.add_argument(
        "--scan-reference-threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold for building a reference hard mask in scan diagnostics, "
            "e.g. --scan-reference-col score --scan-reference-threshold 0.8."
        ),
    )
    parser.add_argument(
        "--scan-reference-op",
        type=str,
        choices=("lt", "le", "gt", "ge"),
        default="lt",
        help="Comparison used for --scan-reference-col threshold diagnostics.",
    )
    return parser.parse_args()


def _normalize_01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=series.index)
    return ((values - vmin) / (vmax - vmin)).fillna(0.0).clip(0.0, 1.0)


def _zeros_like(series: pd.Series) -> pd.Series:
    return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rank_normalize_01(series: pd.Series, higher_is_harder: bool = True) -> pd.Series:
    values = _numeric_series(series)
    finite = values.dropna()
    if finite.empty:
        return _zeros_like(series)
    if float(finite.max()) <= float(finite.min()):
        return _zeros_like(series)
    ranking_values = values if higher_is_harder else -values
    ranked = ranking_values.rank(method="average", pct=True).fillna(0.0).astype(np.float32)
    return ranked.clip(0.0, 1.0)


def _pick_column(df: pd.DataFrame, primary: str, fallback: Optional[str] = None) -> Optional[str]:
    if primary and primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _build_reference_mask(df: pd.DataFrame, args: argparse.Namespace) -> Optional[pd.Series]:
    ref_bool_col = str(args.scan_reference_bool_col).strip()
    if ref_bool_col:
        if ref_bool_col not in df.columns:
            raise ValueError(f"Missing scan reference bool column: {ref_bool_col}")
        ref = pd.to_numeric(df[ref_bool_col], errors="coerce").fillna(0.0) > 0.5
        return ref.astype(bool)

    if args.scan_reference_threshold is None:
        return None

    ref_col = _pick_column(df, args.scan_reference_col, fallback="score")
    if ref_col is None:
        raise ValueError(f"Missing scan reference column: {args.scan_reference_col}")
    values = pd.to_numeric(df[ref_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    thr = float(args.scan_reference_threshold)
    op = str(args.scan_reference_op).strip().lower()
    if op == "lt":
        ref = values < thr
    elif op == "le":
        ref = values <= thr
    elif op == "gt":
        ref = values > thr
    else:
        ref = values >= thr
    return ref.fillna(False).astype(bool)


def _build_scan_rows(
    df: pd.DataFrame,
    fx: pd.Series,
    args: argparse.Namespace,
    plan_col: str,
) -> List[dict]:
    target_rates = sorted(set(_parse_fraction_list(args.scan_target_hard_rates)))
    reference_mask = _build_reference_mask(df, args)
    score_col = "score" if "score" in df.columns else plan_col
    score_values = pd.to_numeric(df[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fx_np = fx.to_numpy(dtype=np.float64)
    rows: List[dict] = []

    for rate in target_rates:
        rate = float(min(max(rate, 1e-8), 1.0))
        q = 1.0 - rate
        thr = float(np.quantile(fx_np, q))
        pred_mask = (fx >= thr).astype(bool)
        actual_rate = float(pred_mask.mean())
        row = {
            "target_hard_rate": rate,
            "thr_by_quantile": thr,
            "actual_hard_rate": actual_rate,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_score_hard": float(score_values[pred_mask].mean(skipna=True))
            if bool(pred_mask.any())
            else np.nan,
        }
        if reference_mask is not None:
            tp = int((pred_mask & reference_mask).sum())
            fp = int((pred_mask & (~reference_mask)).sum())
            fn = int(((~pred_mask) & reference_mask).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if (precision + recall) <= 0.0 else 2.0 * precision * recall / (precision + recall)
            row["precision"] = precision
            row["recall"] = recall
            row["f1"] = f1
        rows.append(row)
    return rows


def _build_regret_signal(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.Series, str]:
    explicit_col = str(args.regret_col).strip()
    if explicit_col:
        col = _pick_column(df, explicit_col)
        if col is None:
            raise ValueError(f"Missing regret column: {explicit_col}")
        return _numeric_series(df[col]).fillna(0.0).clip(lower=0.0), col

    best_col = _pick_column(df, "hc_best_mode_oracle_score")
    selected_col = _pick_column(df, "hc_selected_mode_oracle_score")
    if best_col is not None and selected_col is not None:
        best = _numeric_series(df[best_col])
        selected = _numeric_series(df[selected_col])
        regret = (best - selected).fillna(0.0).clip(lower=0.0)
        return regret, f"{best_col}-{selected_col}"

    percentile_col = _pick_column(df, "hc_selected_mode_oracle_percentile")
    if percentile_col is not None:
        percentile = _numeric_series(df[percentile_col]).fillna(0.0).clip(0.0, 1.0)
        regret = (1.0 - percentile).clip(0.0, 1.0)
        return regret, f"1-{percentile_col}"

    rank_col = _pick_column(df, "hc_selected_mode_oracle_rank")
    if rank_col is not None:
        rank = _numeric_series(df[rank_col]).fillna(1.0).clip(lower=1.0)
        regret = (rank - 1.0).clip(lower=0.0)
        return regret, f"{rank_col}-1"

    raise ValueError(
        "regret_risk mode requires one of: --regret-col, "
        "hc_best_mode_oracle_score + hc_selected_mode_oracle_score, "
        "hc_selected_mode_oracle_percentile, or hc_selected_mode_oracle_rank."
    )


def main() -> None:
    args = _parse_args()
    if not args.score_csv.exists():
        raise FileNotFoundError(f"score csv not found: {args.score_csv}")

    df = pd.read_csv(args.score_csv)
    if "token" not in df.columns:
        raise ValueError("CSV must contain token column")
    df = df[df["token"].astype(str) != "average"].copy()
    if not args.include_invalid and "valid" in df.columns:
        valid = pd.to_numeric(df["valid"], errors="coerce").fillna(0.0) > 0.5
        df = df[valid].copy()
    if len(df) == 0:
        raise ValueError("No valid rows after filtering.")

    hardness_mode = str(args.hardness_mode).strip().lower()
    plan_col = _pick_column(df, args.plan_col, fallback="score")
    if plan_col is None:
        raise ValueError(f"Missing F_plan column: {args.plan_col}")
    f_plan = pd.to_numeric(df[plan_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    analysis_cols: List[str] = []

    if hardness_mode == "legacy":
        per_col = _pick_column(df, args.per_col) if args.per_col else None
        if per_col is None:
            f_per = _zeros_like(df[plan_col])
        else:
            f_per = _normalize_01(df[per_col])

        bev_col = _pick_column(df, args.bev_col) if args.bev_col else None
        if bev_col is None:
            f_bev = _zeros_like(df[plan_col])
        else:
            f_bev = _normalize_01(df[bev_col])

        ent_col = _pick_column(df, args.ent_col, fallback="hc_u_adapter_variance_raw")
        if ent_col is None:
            f_ent = _zeros_like(df[plan_col])
        else:
            f_ent = _normalize_01(df[ent_col])

        fx = (
            (1.0 - f_plan)
            + float(args.beta_per) * f_per
            + float(args.beta_bev) * f_bev
            + float(args.beta_ent) * f_ent
        ).astype(np.float32)
        regret_col = None
        regret_source = None
        attn_col = None
        score_var_col = None
        df["r2se_plan_rank"] = _rank_normalize_01(1.0 - f_plan)
        analysis_cols.extend(["r2se_plan_rank"])
    else:
        per_col = None
        ent_col = None
        bev_col = _pick_column(df, args.bev_col) if args.bev_col else None
        attn_col = _pick_column(df, args.attn_col) if args.attn_col else None
        score_var_col = _pick_column(df, args.score_var_col) if args.score_var_col else None
        if (
            float(args.beta_regret) <= 0.0
            and float(args.beta_bev) <= 0.0
            and float(args.beta_attn) <= 0.0
            and float(args.beta_var) <= 0.0
            and float(args.beta_plan_rank) <= 0.0
        ):
            raise ValueError("regret_risk mode requires at least one positive weight.")
        regret_signal, regret_source = _build_regret_signal(df, args)
        regret_col = str(args.regret_col).strip() or regret_source

        regret_rank = _rank_normalize_01(regret_signal)
        bev_rank = _rank_normalize_01(df[bev_col]) if bev_col is not None else _zeros_like(df[plan_col])
        attn_rank = _rank_normalize_01(df[attn_col]) if attn_col is not None else _zeros_like(df[plan_col])
        score_var_rank = (
            _rank_normalize_01(df[score_var_col]) if score_var_col is not None else _zeros_like(df[plan_col])
        )
        plan_rank = _rank_normalize_01(1.0 - f_plan)

        df["r2se_regret"] = regret_signal.astype(np.float32)
        df["r2se_regret_rank"] = regret_rank
        df["r2se_plan_rank"] = plan_rank
        df["r2se_bev_rank"] = bev_rank
        df["r2se_attn_rank"] = attn_rank
        df["r2se_score_var_rank"] = score_var_rank
        analysis_cols.extend(
            [
                "r2se_regret",
                "r2se_regret_rank",
                "r2se_plan_rank",
                "r2se_bev_rank",
                "r2se_attn_rank",
                "r2se_score_var_rank",
            ]
        )

        fx = (
            float(args.beta_regret) * regret_rank
            + float(args.beta_bev) * bev_rank
            + float(args.beta_attn) * attn_rank
            + float(args.beta_var) * score_var_rank
            + float(args.beta_plan_rank) * plan_rank
        ).astype(np.float32)
    eps_percent = float(args.eps_percent)
    eps_percent = min(max(eps_percent, 1e-6), 100.0)
    q = 1.0 - eps_percent / 100.0
    threshold = float(np.quantile(fx.to_numpy(), q))
    target_topk = int(np.ceil(len(df) * (eps_percent / 100.0)))
    target_topk = max(min(target_topk, len(df)), 1)

    if bool(args.strict_topk):
        work_df = df.copy()
        work_df["r2se_fx"] = fx
        work_df["_row_order"] = np.arange(len(work_df), dtype=np.int64)
        hard_df = (
            work_df.sort_values(
                ["r2se_fx", "_row_order"], ascending=[False, True], kind="mergesort"
            )
            .head(target_topk)
            .drop(columns=["_row_order"])
            .reset_index(drop=True)
        )
        effective_threshold = float(hard_df["r2se_fx"].iloc[-1])
    else:
        hard_mask = fx >= threshold
        hard_df = df.loc[hard_mask].copy()
        hard_df["r2se_fx"] = fx.loc[hard_mask]
        hard_df = hard_df.sort_values("r2se_fx", ascending=False).reset_index(drop=True)
        effective_threshold = threshold

    output = {
        "source_csv": str(args.score_csv),
        "hardness_mode": hardness_mode,
        "eps_percent": eps_percent,
        "quantile": q,
        "threshold": threshold,
        "effective_threshold": effective_threshold,
        "strict_topk": bool(args.strict_topk),
        "target_topk": int(target_topk),
        "plan_col": plan_col,
        "per_col": per_col,
        "bev_col": bev_col,
        "ent_col": ent_col,
        "regret_col": regret_col,
        "regret_source": regret_source,
        "attn_col": attn_col,
        "score_var_col": score_var_col,
        "beta_per": float(args.beta_per),
        "beta_bev": float(args.beta_bev),
        "beta_ent": float(args.beta_ent),
        "beta_regret": float(args.beta_regret),
        "beta_attn": float(args.beta_attn),
        "beta_var": float(args.beta_var),
        "beta_plan_rank": float(args.beta_plan_rank),
        "num_rows": int(len(df)),
        "num_hard": int(len(hard_df)),
        "hard_tokens": [str(t) for t in hard_df["token"].tolist()],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["token", "r2se_fx", plan_col]
        if per_col is not None:
            cols.append(per_col)
        if bev_col is not None:
            cols.append(bev_col)
        if ent_col is not None:
            cols.append(ent_col)
        for extra in analysis_cols:
            if extra in hard_df.columns and extra not in cols:
                cols.append(extra)
        for extra in (
            "score",
            "hc_bev_ce",
            "hc_attn_aux",
            "score_mode_var_generalist",
            "hc_selected_mode_oracle_rank",
            "hc_selected_mode_oracle_percentile",
            "hc_selected_mode_oracle_score",
            "hc_best_mode_oracle_score",
            "hc_u_adapter_variance",
        ):
            if extra in hard_df.columns and extra not in cols:
                cols.append(extra)
        hard_df[cols].to_csv(args.out_csv, index=False)

    if args.scan_out_csv is not None:
        scan_rows = _build_scan_rows(df=df, fx=fx, args=args, plan_col=plan_col)
        scan_df = pd.DataFrame(scan_rows)
        args.scan_out_csv.parent.mkdir(parents=True, exist_ok=True)
        scan_df.to_csv(args.scan_out_csv, index=False)

    print(
        f"[R2SE hardset] rows={len(df)} hard={len(hard_df)} "
        f"eps={eps_percent:.4f}% threshold={threshold:.6f}"
    )
    print(f"[R2SE hardset] json: {args.out_json}")
    if args.out_csv is not None:
        print(f"[R2SE hardset] csv : {args.out_csv}")
    if args.scan_out_csv is not None:
        print(f"[R2SE hardset] scan: {args.scan_out_csv}")


if __name__ == "__main__":
    main()
