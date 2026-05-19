import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


DEFAULT_RESIDUAL_PREFIXES = (
    "_transfuser_model._trajectory_head._pdm_score_head._score_res_base_proj.",
    "_transfuser_model._trajectory_head._pdm_score_head._score_res_scene_proj.",
    "_transfuser_model._trajectory_head._pdm_score_head._score_res_proposal_proj.",
    "_transfuser_model._trajectory_head._pdm_score_head._score_res_mode_context_proj.",
    "_transfuser_model._trajectory_head._pdm_score_head._score_residual_head.",
    "_transfuser_model._trajectory_head._pdm_score_head._score_residual_lora_adapters.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebase a score-residual/LoRA specialist checkpoint onto a base checkpoint. "
            "All non-residual parameters and buffers are copied from the base checkpoint, "
            "while residual-score modules stay from the specialist checkpoint."
        )
    )
    parser.add_argument("--base-ckpt", required=True, type=Path)
    parser.add_argument("--specialist-ckpt", required=True, type=Path)
    parser.add_argument("--out-ckpt", required=True, type=Path)
    parser.add_argument(
        "--residual-prefix",
        action="append",
        default=None,
        help=(
            "Optional additional residual key prefix. Can be passed multiple times. "
            "Prefixes are matched after stripping an optional leading 'agent.'."
        ),
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a state_dict")
    return checkpoint


def _normalize_key(key: str) -> str:
    return key[len("agent.") :] if key.startswith("agent.") else key


def _is_residual_key(key: str, prefixes: Iterable[str]) -> bool:
    normalized = _normalize_key(key)
    return any(normalized.startswith(prefix) for prefix in prefixes)


def main() -> None:
    args = _parse_args()
    prefixes: List[str] = list(DEFAULT_RESIDUAL_PREFIXES)
    if args.residual_prefix:
        prefixes.extend(args.residual_prefix)

    base_ckpt = _load_checkpoint(args.base_ckpt)
    specialist_ckpt = _load_checkpoint(args.specialist_ckpt)

    base_state = base_ckpt["state_dict"]
    specialist_state = specialist_ckpt["state_dict"]
    merged_ckpt = copy.deepcopy(specialist_ckpt)
    merged_state = merged_ckpt["state_dict"]

    copied_from_base = 0
    kept_from_specialist = 0
    missing_in_base: List[str] = []

    for key, value in specialist_state.items():
        if _is_residual_key(key, prefixes):
            kept_from_specialist += 1
            continue
        if key in base_state:
            merged_state[key] = base_state[key]
            copied_from_base += 1
        else:
            missing_in_base.append(key)

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_ckpt, args.out_ckpt)

    print(f"saved: {args.out_ckpt}")
    print(f"copied_from_base: {copied_from_base}")
    print(f"kept_from_specialist: {kept_from_specialist}")
    if missing_in_base:
        print(f"missing_in_base: {len(missing_in_base)}")
        for key in missing_in_base[:50]:
            print(f"  {key}")


if __name__ == "__main__":
    main()
