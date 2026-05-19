#!/bin/bash
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --nodelist=atl1-1-01-010-31-0,atl1-1-01-010-35-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_camdrop_train
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_camdrop_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_camdrop_%j.err
#
# LoRA finetune of pi0.5 + Adapt3R on the 2-task MimicGen Mesa bimanual dataset
# (4 cams, 200 demos per task) with 3D positional encoding AND per-sample,
# single-camera attention-mask dropout (p=0.25). TrainConfig:
# pi05_mesa_bimanual_lora_3d_gen_camdrop. Prereq: norm stats at
#   assets/pi05_mesa_bimanual_lora_3d_gen_camdrop/mesa_bimanual_gen_2task_3d/
# (symlink to the base 3d_gen config is fine — identical data transforms.)
#
# Rely on --resume to continue across submissions until num_train_steps
# (30_000) is hit.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   EXP_NAME   : experiment name (default: mesa_bimanual_lora_3d_gen_camdrop_v1)
#   BATCH_SIZE : per-device batch (default: 32; L40S has 48 GB VRAM)

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=pair-diffusion

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_3d_gen_camdrop_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-3d-gen-camdrop] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-3d-gen-camdrop] config=pi05_mesa_bimanual_lora_3d_gen_camdrop exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-3d-gen-camdrop] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-3d-gen-camdrop] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_camdrop/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_camdrop \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-3d-gen-camdrop] done."
