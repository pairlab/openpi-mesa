#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_2d_gen_camdrop_train_h100
#SBATCH -o examples/mesa/slurm_logs/train_2d_gen_camdrop_h100_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_2d_gen_camdrop_h100_%j.err
#
# H100/embers variant of sbatch_train_2d_gen_camdrop.sh — same training config
# (pi05_mesa_bimanual_lora_2d_gen_camdrop, batch 32, --resume), routed to the
# gpu-h100 partition under the embers QOS instead of gpu-l40s/inferno.
# Used to keep training advancing past 10k while the L40S/inferno queue is
# saturated. Relies on --resume to pick up from the latest checkpoint and
# continue until num_train_steps (30_000) is hit.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   EXP_NAME   : experiment name (default: mesa_bimanual_lora_2d_gen_camdrop_v1)
#   BATCH_SIZE : per-device batch (default: 32; H100 has 80 GB VRAM, plenty of headroom)

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

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_2d_gen_camdrop_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-2d-gen-camdrop-h100] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-2d-gen-camdrop-h100] config=pi05_mesa_bimanual_lora_2d_gen_camdrop exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-2d-gen-camdrop-h100] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-2d-gen-camdrop-h100] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_2d_gen_camdrop/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_2d_gen_camdrop \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-2d-gen-camdrop-h100] done."
