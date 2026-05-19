#!/bin/bash
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_2d_gen_ego_train_inferno
#SBATCH -o examples/mesa/slurm_logs/train_2d_gen_ego_inferno_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_2d_gen_ego_inferno_%j.err
#
# L40S/inferno variant of sbatch_train_2d_gen_ego.sh — same training config
# (pi05_mesa_bimanual_lora_2d_gen_ego, batch 32, --resume), routed to the
# project's inferno QOS instead of embers. Use this when embers H100/L40S
# slots stay queued and the project allocation has L40S capacity.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache_l40s"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=pair-diffusion

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_2d_gen_ego_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-2d-gen-ego-inferno] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-2d-gen-ego-inferno] config=pi05_mesa_bimanual_lora_2d_gen_ego exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-2d-gen-ego-inferno] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-2d-gen-ego-inferno] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_2d_gen_ego/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_2d_gen_ego \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-2d-gen-ego-inferno] done."
