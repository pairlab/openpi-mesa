#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_train
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_%j.err
#
# LoRA finetune of pi0.5 on the 2-task MimicGen-generated Mesa bimanual LeRobot
# dataset (4 cams, 200 demos per task) with 3D positional encoding. TrainConfig:
# pi05_mesa_bimanual_lora_3d_gen. Prereq: norm stats at
#   assets/pi05_mesa_bimanual_lora_3d_gen/mesa_bimanual_gen_2task_3d/norm_stats.json
#
# Embers walltime is hard-capped at 8h regardless of -t; rely on --resume to
# continue across submissions until num_train_steps (30_000) is hit.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   EXP_NAME   : experiment name (default: mesa_bimanual_lora_3d_gen_v1)
#   BATCH_SIZE : per-device batch (default: 32; L40S has 48 GB VRAM)

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_ENTITY=pair-diffusion

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_3d_gen_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-3d-gen] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-3d-gen] config=pi05_mesa_bimanual_lora_3d_gen exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-3d-gen] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-3d-gen] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-3d-gen] done."
