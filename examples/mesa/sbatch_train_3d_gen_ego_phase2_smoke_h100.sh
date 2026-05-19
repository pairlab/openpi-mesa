#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_ego_phase2_smoke_h100
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_ego_phase2_smoke_h100_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_ego_phase2_smoke_h100_%j.err
#
# H100/embers sibling of sbatch_train_3d_gen_ego_phase2_smoke_inferno.sh —
# raced against the inferno/l40s submission so whichever queue frees first
# runs the smoke. Cancel the loser once one transitions to RUNNING (same
# pattern as submit_rope_ego_eval.sh). The JAX_COMPILATION_CACHE_DIR is
# partitioned per GPU type (jax_cache_h100 vs jax_cache_l40s) since cross-GPU
# caches silently SIGSEGV at first JIT step.
#
# Validates the same things as the inferno smoke: weight loading from
# pi05_base with the per-arm + temporal_pos_embed carve-out, extra_arms
# data wiring, and JIT compile + a few diffusion-loss steps without
# NaN/Inf. Success criterion: monotone-ish loss decrease and no Traceback.
#
# Prereq: norm_stats at
#   assets/pi05_mesa_bimanual_lora_3d_gen_ego_phase2/mesa_bimanual_gen_2task_3d/
# (symlinked to assets/pi05_mesa_bimanual_lora_3d_gen/... since the action and
# state schemas are unchanged from Phase 1).

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache_h100"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

EXP_NAME="${EXP_NAME:-phase2_smoke}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-50}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[phase2-smoke-h100] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[phase2-smoke-h100] config=pi05_mesa_bimanual_lora_3d_gen_ego_phase2 exp=$EXP_NAME batch=$BATCH_SIZE steps=$NUM_STEPS"
echo "[phase2-smoke-h100] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_ego_phase2/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_ego_phase2 \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_STEPS" \
    --no-wandb-enabled \
    --overwrite

echo "[phase2-smoke-h100] done."
