from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a token subset from a pdm score csv for diagnostic reruns."
    )
    parser.add_argument("--score-csv", required=True, help="Input pdm score csv path")
    parser.add_argument("--out-json", required=True, help="Output json path")
    parser.add_argument(
        "--score-max",
        type=float,
        default=None,
        help="Keep rows with score <= score-max",
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=None,
        help="Keep rows with score >= score-min",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Require valid == True when filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.score_csv)
    if "token" not in df.columns or "score" not in df.columns:
        raise ValueError("Input csv must contain 'token' and 'score' columns")

    filtered = df.copy()
    filtered = filtered[filtered["token"] != "average"]
    if args.valid_only and "valid" in filtered.columns:
        filtered = filtered[filtered["valid"].fillna(False).astype(bool)]
    if args.score_max is not None:
        filtered = filtered[filtered["score"] <= args.score_max]
    if args.score_min is not None:
        filtered = filtered[filtered["score"] >= args.score_min]

    tokens = filtered["token"].dropna().astype(str).tolist()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": tokens,
        "count": len(tokens),
        "score_csv": str(Path(args.score_csv)),
        "score_max": args.score_max,
        "score_min": args.score_min,
        "valid_only": bool(args.valid_only),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[extract_tokens_by_score] tokens={len(tokens)} json={out_path}")


if __name__ == "__main__":
    main()
