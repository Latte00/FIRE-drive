# FIRE-drive

Minimal test project for the FIRE-drive / DiffusionDrive NAVSIM agent.

This repository intentionally does not include the trained checkpoint. Put the checkpoint at:

```text
checkpoints/best_pdm_riskattn.ckpt
```

or pass another path through `CKPT` / `-Ckpt`.

## Contents

- `navsim/`: NAVSIM evaluation code plus the `navsim.agents.diffusiondrive` model used by the checkpoint.
- `configs/training_run/`: Hydra config snapshot from `pdm_online_drivor_style_riskattn/2026.04.18.11.39.54`.
- `assets/kmeans_navsim_traj_20.npy`: trajectory anchor file required at model construction.
- `scripts/evaluation/`: shell and PowerShell wrappers for PDM evaluation with the checkpoint's architecture overrides.

The original `best_pdm_riskattn.ckpt` is about 761 MB and is ignored by Git.

## Setup

```bash
conda create -n fire-drive python=3.9 -y
conda activate fire-drive
pip install -e .
pip install diffusers einops
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
CKPT=checkpoints/best_pdm_riskattn.ckpt bash scripts/evaluation/run_fire_drive_eval.sh
```

Windows PowerShell:

```powershell
.\scripts\evaluation\run_fire_drive_eval.ps1 -Ckpt "checkpoints\best_pdm_riskattn.ckpt"
```

If `assets/resnet34.a1_in1k/pytorch_model.bin` is absent, the copied backbone initializes with `pretrained=False`; loading the FIRE-drive checkpoint supplies the trained weights.
