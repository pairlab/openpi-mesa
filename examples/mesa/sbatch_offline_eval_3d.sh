#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J pi05_mesa_bimanual_lora_3d_offline_eval
#SBATCH -o examples/mesa/slurm_logs/offline_eval_3d_%j.out
#SBATCH -e examples/mesa/slurm_logs/offline_eval_3d_%j.err
#
# Offline prediction check for a trained pi05_mesa_bimanual_lora_3d checkpoint:
# runs model.sample_actions on a few training batches and reports how close
# predictions are to ground truth. Overall MSE near zero on training data is
# the overfitting signal we want before investing in more training.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export GIT_LFS_SKIP_SMUDGE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/pi05_mesa_bimanual_lora_3d/mesa_bimanual_lora_3d_v1/7000}"

echo "[offline-eval] node=$(hostname) cpu=$(nproc) platform=cpu-only"
echo "[offline-eval] checkpoint=$CHECKPOINT_DIR"

uv run python examples/mesa/offline_eval_3d.py \
    --config-name pi05_mesa_bimanual_lora_3d \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size 4 \
    --num-batches 3 \
    --num-sample-steps 10

echo "[offline-eval] done."
