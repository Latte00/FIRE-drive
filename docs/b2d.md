# Bench2Drive

Use the repository's Python dependencies, CARLA 0.9.15 (matching Python client),
Bench2Drive and Bench2DriveZoo. The latter supplies `team_code.planner` and
`team_code.pid_controller`; these external components retain their upstream
licenses. Use CPython, not GraalPy. Verify NVIDIA Vulkan rendering before evaluation.

## Inference

Fetch `checkpoints/firedrive_b2d_navigation.ckpt` with Git LFS. This is a
weights-only export; it does not contain optimizer state or private training metadata.

```bash
export CARLA_ROOT=/path/to/carla
export Bench2Drive_ROOT=/path/to/Bench2Drive
export B2D_ZOO_ROOT=/path/to/Bench2DriveZoo
GPU_RANK=0 OUT="$PWD/outputs/b2d_navigation" bash scripts/b2d/evaluate.sh
```

The launcher uses the Bench2Drive evaluator's server lifecycle and resumes
`OUT/results.json`. Use a new `OUT` for each model/configuration. For a smoke
test, set `ROUTES_SUBSET=24211`. Concurrent runs need different `PORT` and
`TM_PORT`. `CFG` and `CKPT` can override the defaults.

The online configuration uses Epic rendering, a three-camera panorama
(`b2d_mid`, side crop/overlap `0.30/0.17`, JPEG quality 20), front-left LiDAR
coordinates, two motion-compensated half-scans (10 Hz, 600k points/s), and the
trajectory-only AutoMoT-style controller. Near/far route points condition the
model; they do not replace its steering aim. Model weights are loaded strictly.

## Training

Create the standard B2D cache with the existing
`navsim.planning.script.run_b2d_diffusiondrive_caching` tool (`--help` lists data
and infos paths). Cache and online panorama defaults are both `b2d_mid` with
side crop/overlap `0.30/0.17`. This reconstruction matched a checked training
cache sample pixel-for-pixel; other samples should be checked for mixed caches.
For explicit cache generation, pass `--camera-stitch-mode b2d_mid
--camera-side-crop-ratio 0.30 --camera-side-overlap-ratio 0.17`.
Keep 6 future poses at 0.5 s, histogram LiDAR, and lane-direction supervision
(`--write-lane-direction-bev`). Navigation fields are written automatically.
For an older cache, add navigation sidecars without rewriting sensor tensors:

```bash
python -m navsim.planning.script.patch_b2d_navigation_features \
  --infos-root /path/to/infos --cache-root /path/to/cache --splits train val
export B2D_CACHE_ROOT=/path/to/cache
CUDA_VISIBLE_DEVICES=0 bash scripts/b2d/train.sh
```

The supplied recipe matches full-data training: `run_training_full` trains on
**train + val**, and monitors val for checkpoint selection. This is not a
held-out validation protocol. It saves best and last checkpoints. The optional
ImageNet backbone file is `assets/resnet34.a1_in1k/pytorch_model.bin`; without
it, training starts that backbone randomly. Inference restores the full backbone
from the supplied checkpoint. NAVSIM defaults and entry points are unchanged;
navigation remains disabled unless explicitly enabled.
