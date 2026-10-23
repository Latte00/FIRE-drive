#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${B2D_CACHE_ROOT:?Set B2D_CACHE_ROOT to a cache containing train/ and val/}"
[[ -d "$B2D_CACHE_ROOT/train" && -d "$B2D_CACHE_ROOT/val" ]] || { echo "Missing train/val cache" >&2; exit 1; }
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"$ROOT/outputs"}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$ROOT"
python -m navsim.planning.script.run_training_full \
    --config-path "$ROOT/configs/b2d" --config-name training "$@"
