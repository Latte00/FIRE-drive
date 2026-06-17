# FIRE-drive

Minimal test project for the FIRE-drive / DiffusionDrive NAVSIM agent.

This repository includes the FIRE-drive inference checkpoint through Git LFS:

```text
checkpoints/FireDrive_riskattn64_single_gpu-915.ckpt
```

or pass another path through `CKPT` / `-Ckpt`.

## Contents

- `navsim/`: NAVSIM evaluation code plus the `navsim.agents.diffusiondrive` model used by the checkpoint.
- `configs/training_run/`: Hydra config snapshot from `pdm_online_drivor_style_riskattn/2026.04.18.11.39.54`.
- `assets/kmeans_navsim_traj_20.npy`: trajectory anchor file required at model construction.
- `scripts/evaluation/`: shell and PowerShell wrappers for PDM evaluation with the checkpoint's architecture overrides.
- `checkpoints/FireDrive_riskattn64_single_gpu-915.ckpt`: Git LFS checkpoint used by the default evaluation scripts.

The larger full-training checkpoint is not included; it contains training state that is not needed for evaluation.

## Baseline

This project is based on the official DiffusionDrive baseline:

- [hustvl/DiffusionDrive](https://github.com/hustvl/diffusiondrive)

## Benchmark

NAVSIM benchmark results:

| Method | Backbone | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transfuser (2022) | ResNet34 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| UniAD (2023) | ResNet34 | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| LAW (2024) | ResNet34 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DRAMA (2024) | ResNet34 | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| Hydra-MDP (2024) | ResNet34 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| Hydra-MDP++ (2025) | ResNet34 | 97.6 | 96.0 | 80.4 | 93.1 | 100 | 86.6 |
| GoalFlow (2025) | ResNet34 | 98.3 | 93.8 | 79.8 | 94.3 | 100 | 85.7 |
| DiffusionDrive (2025) | ResNet34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| DiffusionDrive V2 (2025) | ResNet34 | 98.3 | 97.9 | 87.5 | 94.8 | 99.9 | 91.2 |
| DIVER (2025) | ResNet34 | 98.5 | 96.5 | 82.6 | 94.9 | 100 | 88.3 |
| RESAD (2025) | ResNet34 | 98.0 | 97.5 | 83.3 | 94.1 | 100 | 88.8 |
| DriveSuprim (2026) | ResNet34 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 |
| **FIRE-Drive (Ours)** | **ResNet34** | **98.9** | **97.5** | **87.6** | **95.8** | **100** | **91.5** |
| Hydra-MDP (2024) | V2-99 | 98.4 | 97.8 | 86.5 | 93.9 | 100 | 90.3 |
| Hydra-MDP (2024) | ViT-L | 98.4 | 97.7 | 85.0 | 94.5 | 100 | 89.9 |
| RESAD (2025) | V2-99 | 98.9 | 97.8 | 87.0 | 94.9 | 100 | 90.6 |
| Hydra-MDP++ (2025) | V2-99 | 98.6 | 98.6 | 85.7 | 95.1 | 100 | 91.0 |
| VADv2 (2026) | BEVFormer | 98.3 | 97.4 | 82.3 | 95.7 | 100 | 89.3 |
| Human Agent | - | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |

## Setup

```bash
conda create -n fire-drive python=3.9 -y
conda activate fire-drive
pip install -e .
pip install diffusers einops
```

If you clone this repository elsewhere, install Git LFS and pull the checkpoint:

```bash
git lfs install
git lfs pull
```

Set the NAVSIM data/output roots before evaluation:

```bash
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NAVSIM_EXP_ROOT=/path/to/navsim_exp
```

`OPENSCENE_DATA_ROOT` should contain `navsim_logs/` and `sensor_blobs/` for the selected split.

## Prepare Metric Cache

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
```

## Evaluate

Linux/macOS:

```bash
bash scripts/evaluation/run_fire_drive_eval.sh
```

Windows PowerShell:

```powershell
.\scripts\evaluation\run_fire_drive_eval.ps1
```

If `assets/resnet34.a1_in1k/pytorch_model.bin` is absent, the copied backbone initializes with `pretrained=False`; loading the FIRE-drive checkpoint supplies the trained weights.
