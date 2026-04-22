#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_lora_train
#SBATCH -o examples/mesa/slurm_logs/train_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_%j.err
#
# LoRA finetune of pi0.5 on the 2-task Mesa bimanual LeRobot dataset
# (TrainConfig: pi05_mesa_bimanual_lora, 30_000 steps, batch_size=32).
# Prereq: norm stats at
#   assets/pi05_mesa_bimanual_lora/fchang40/mesa_bimanual_2task/norm_stats.json
#
# Checkpoints land off-home on the project volume; home quota is 20G.
# Base weights (gs://openpi-assets/checkpoints/pi05_base/params) are cached
# off-home via OPENPI_DATA_HOME, and the JAX compile cache is relocated via
# JAX_COMPILATION_CACHE_DIR for the same reason.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0"
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

EXP_NAME="${EXP_NAME:-pi05_bimanual_2task_lora_v2}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train] config=pi05_mesa_bimanual_lora exp=$EXP_NAME"
echo "[train] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR"

echo "[train] done."
