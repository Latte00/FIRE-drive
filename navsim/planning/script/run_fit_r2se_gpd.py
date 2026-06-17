from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

try:
    from scipy.stats import genpareto
except Exception:  # pragma: no cover
    genpareto = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit GPD tail for R2SE uncertainty routing from run_pdm_score CSV "
            "and optional hardset token list."
        )
    )
    parser.add_argument("--score-csv", type=Path, required=True, help="Input run_pdm_score csv.")
    parser.add_argument(
        "--hardset-json",
        type=Path,
        default=None,
        help="Optional hardset json from run_build_r2se_hardset.py",
    )
    parser.add_argument(
        "--u-col",
        type=str,
        default="hc_u_adapter_variance",
        help="Uncertainty column used to fit GPD tail.",
    )
    parser.add_argument(
        "--u0-quantile",
        type=float,
        default=0.80,
        help="Threshold quantile u0 for tail fitting.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.75,
        help="Switch threshold sigma used in Eq.14-style routing.",
    )
    parser.add_argument(
        "--min-tail-samples",
        type=int,
        default=20,
        help="Minimum samples required in tail; otherwise fallback to exponential (shape=0).",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Output GPD parameter json.")
    return parser.parse_args()


def _gpd_cdf(u: np.ndarray, u0: float, shape: float, scale: float) -> np.ndarray:
    x = np.maximum(u - u0, 0.0)
    if abs(shape) < 1e-8:
        return np.clip(1.0 - np.exp(-x / max(scale, 1e-6)), 0.0, 1.0)
    if shape < 0.0:
        base = 1.0 + shape * x / max(scale, 1e-6)
        cdf = np.ones_like(x, dtype=np.float64)
        valid = base > 0.0
        cdf[valid] = 1.0 - np.power(base[valid], -1.0 / shape)
        return np.clip(cdf, 0.0, 1.0)
    base = np.maximum(1.0 + shape * x / max(scale, 1e-6), 1e-8)
    return np.clip(1.0 - np.power(base, -1.0 / shape), 0.0, 1.0)


def _load_hard_tokens(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"hardset json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("hard_tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"hard_tokens missing in {path}")
    return {str(t) for t in tokens}


def main() -> None:
    args = _parse_args()
    if not args.score_csv.exists():
        raise FileNotFoundError(f"score csv not found: {args.score_csv}")

    df = pd.read_csv(args.score_csv)
    if "token" not in df.columns:
        raise ValueError("CSV must contain token column")
    df = df[df["token"].astype(str) != "average"].copy()
    if "valid" in df.columns:
        valid_mask = pd.to_numeric(df["valid"], errors="coerce").fillna(0.0) > 0.5
        df = df[valid_mask].copy()

    hard_tokens = _load_hard_tokens(args.hardset_json)
    if hard_tokens is not None:
        df = df[df["token"].astype(str).isin(hard_tokens)].copy()
    if len(df) == 0:
        raise ValueError("No rows available for GPD fitting after filtering.")

    u_col = args.u_col
    if u_col not in df.columns:
        if "hc_u_adapter_variance" in df.columns:
            u_col = "hc_u_adapter_variance"
        elif "hc_u_adapter_variance_raw" in df.columns:
            u_col = "hc_u_adapter_variance_raw"
        else:
            raise ValueError(f"Uncertainty column not found: {args.u_col}")
    u = pd.to_numeric(df[u_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(u) == 0:
        raise ValueError(f"No finite uncertainty values in column {u_col}")
    u_np = u.to_numpy(dtype=np.float64)

    q = float(min(max(args.u0_quantile, 0.0), 0.999999))
    u0 = float(np.quantile(u_np, q))
    excess = u_np[u_np >= u0] - u0
    excess = excess[np.isfinite(excess)]
    if excess.size <= 0:
        raise ValueError("No tail samples available after thresholding.")

    shape = 0.0
    scale = float(max(np.mean(excess), 1e-3))
    fit_method = "exponential_fallback"
    if genpareto is not None and excess.size >= int(max(args.min_tail_samples, 1)):
        try:
            c, loc, s = genpareto.fit(excess, floc=0.0)
            if np.isfinite(c) and np.isfinite(s) and s > 0:
                shape = float(c)
                scale = float(s)
                fit_method = "scipy_genpareto_mle"
        except Exception:
            fit_method = "exponential_fallback_after_fit_error"

    sigma = float(min(max(args.sigma, 0.0), 1.0))
    pgpd = _gpd_cdf(u_np, u0=u0, shape=shape, scale=scale)
    switch_rate = float(np.mean(pgpd > sigma))

    output = {
        "source_csv": str(args.score_csv),
        "hardset_json": str(args.hardset_json) if args.hardset_json is not None else None,
        "u_col": u_col,
        "u0_quantile": q,
        "u0": u0,
        "shape": shape,
        "scale": scale,
        "sigma": sigma,
        "fit_method": fit_method,
        "num_rows": int(len(df)),
        "num_u": int(u_np.size),
        "num_tail": int(excess.size),
        "switch_rate_at_sigma": switch_rate,
        "pgpd_mean": float(np.mean(pgpd)),
        "pgpd_std": float(np.std(pgpd)),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[R2SE GPD] u_col={u_col} rows={len(df)} u0={u0:.6f} "
        f"shape={shape:.6f} scale={scale:.6f} sigma={sigma:.3f} "
        f"switch_rate={switch_rate:.4f}"
    )
    print(f"[R2SE GPD] json: {args.out_json}")


if __name__ == "__main__":
    main()
