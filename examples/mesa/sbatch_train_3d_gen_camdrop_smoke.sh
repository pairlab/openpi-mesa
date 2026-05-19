#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_camdrop_smoke
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_camdrop_smoke_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_camdrop_smoke_%j.err
#
# Smoke test for pi05_mesa_bimanual_lora_3d_gen_camdrop (4-cam MimicGen config
# with per-sample, single-camera attention-mask dropout p=0.25). Runs a short
# training window to exercise the dropout code path inside `compute_loss`.
#
# Prereq: norm_stats at
#   assets/pi05_mesa_bimanual_lora_3d_gen_camdrop/mesa_bimanual_gen_2task_3d/
# (symlinked to the base config's identical norm_stats since data transforms
# match).

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

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

EXP_NAME="${EXP_NAME:-camdrop_smoke}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-10}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[camdrop-smoke] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[camdrop-smoke] config=pi05_mesa_bimanual_lora_3d_gen_camdrop exp=$EXP_NAME batch=$BATCH_SIZE steps=$NUM_STEPS"
echo "[camdrop-smoke] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_camdrop/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_camdrop \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_STEPS" \
    --no-wandb-enabled \
    --overwrite

echo "[camdrop-smoke] done."
