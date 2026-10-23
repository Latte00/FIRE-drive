#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CARLA_ROOT:?Set CARLA_ROOT to CARLA 0.9.15}"
: "${Bench2Drive_ROOT:?Set Bench2Drive_ROOT}"
: "${B2D_ZOO_ROOT:?Set B2D_ZOO_ROOT to Bench2DriveZoo}"
CFG=${CFG:-"$ROOT/configs/b2d/training.yaml"}
CKPT=${CKPT:-"$ROOT/checkpoints/firedrive_b2d_navigation.ckpt"}
OUT=${OUT:-"$ROOT/outputs/b2d_navigation"}
GPU_RANK=${GPU_RANK:-0}
PORT=${PORT:-20000}
TM_PORT=${TM_PORT:-40000}
[[ -f "$CFG" && -f "$CKPT" ]] || { echo "Missing config or checkpoint" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
CFG=$(realpath "$CFG")
CKPT=$(realpath "$CKPT")
export Bench2Drive_ROOT CARLA_ROOT B2D_ZOO_ROOT GPU_RANK
export PYTHONPATH="$ROOT:$B2D_ZOO_ROOT:$Bench2Drive_ROOT/leaderboard:$Bench2Drive_ROOT/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
export SCENARIO_RUNNER_ROOT="$Bench2Drive_ROOT/scenario_runner"
export LEADERBOARD_ROOT="$Bench2Drive_ROOT/leaderboard"
export CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"
export IS_BENCH2DRIVE=1
export SAVE_PATH="$OUT"
export CARLA_QUALITY_LEVEL=Epic
# Do not inherit experimental sensor/controller overrides from another run.
while IFS= read -r name; do unset "$name"; done < <(compgen -v DIFFUSIONDRIVE_B2D_ || true)
unset B2D_EXTERNAL_CARLA
export DIFFUSIONDRIVE_B2D_BKB_PATH="$ROOT/assets/resnet34.a1_in1k/pytorch_model.bin"
export DIFFUSIONDRIVE_B2D_PLAN_ANCHOR_PATH="$ROOT/assets/kmeans_navsim_traj_20.npy"
export DIFFUSIONDRIVE_B2D_CAMERA_SCALE=0.64
export DIFFUSIONDRIVE_B2D_STITCH_MODE=b2d_mid
export DIFFUSIONDRIVE_B2D_SIDE_CROP_RATIO=0.30
export DIFFUSIONDRIVE_B2D_SIDE_OVERLAP_RATIO=0.17
export DIFFUSIONDRIVE_B2D_JPEG_QUALITY=20
export DIFFUSIONDRIVE_B2D_LIDAR_MODE=histogram
export DIFFUSIONDRIVE_B2D_LIDAR_COORDINATE_MODE=front_left
export DIFFUSIONDRIVE_B2D_LIDAR_ACCUM_FRAMES=2
export DIFFUSIONDRIVE_B2D_LIDAR_MOTION_COMPENSATION=1
export DIFFUSIONDRIVE_B2D_LIDAR_ROTATION_FREQUENCY=10
export DIFFUSIONDRIVE_B2D_LIDAR_POINTS_PER_SECOND=600000
export DIFFUSIONDRIVE_B2D_LIDAR_SENSOR_TICK=0.05
export DIFFUSIONDRIVE_B2D_LIDAR_Z_OFFSET=2.0
export DIFFUSIONDRIVE_B2D_LIDAR_MIN_Z=0.2
export DIFFUSIONDRIVE_B2D_LIDAR_MAX_Z=100.0
export DIFFUSIONDRIVE_B2D_CONTROLLER=automot
export DIFFUSIONDRIVE_B2D_PID_FRAME=pad
export DIFFUSIONDRIVE_B2D_SAVE_IMAGES=0
export DIFFUSIONDRIVE_B2D_SAVE_LIDAR=0
unset DIFFUSIONDRIVE_B2D_CKPT DIFFUSIONDRIVE_B2D_HYDRA_CONFIG
cd -- "$Bench2Drive_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_RANK" python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes="${ROUTES:-leaderboard/data/bench2drive220.xml}" \
    --routes-subset="${ROUTES_SUBSET:-}" --repetitions=1 --track=SENSORS \
    --checkpoint="$OUT/results.json" --debug-checkpoint="$OUT/live_results.txt" \
    --agent="$ROOT/navsim/agents/diffusiondrive/firedrive_b2d_agent.py" \
    --agent-config="$CFG+$CKPT" --debug=0 --record= --resume=True \
    --port="$PORT" --traffic-manager-port="$TM_PORT" --gpu-rank="$GPU_RANK"
