#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_ego_phase2_train
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_ego_phase2_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_ego_phase2_%j.err
#
# Phase 2 of the 3D RoPE work — per-arm action-token split with 3D RoPE keyed
# by per-arm gripper xyz (markdowns/B.md §8, D.md root-cause + fix). Same data
# and budget as Phase 1; the TrainConfig `pi05_mesa_bimanual_lora_3d_gen_ego_phase2`
# instantiates `Pi0Adapt3RBimanualConfig` with `per_arm_action_dim=7` so the
# per-arm slice routes [jp0, grip0] to arm 0 and [jp1, grip1] to arm 1 (the
# default `action_dim // 2 = 16` was wrong for this layout — see
# pi0_adapt3r_test.py).
#
# Embers notes (see ~/.claude memory): use `--gres=gpu:l40s:1` (plain `gpu:1`
# would backfill to V100 via embers multi-partition) and cpus-per-task=4. The
# 8h walltime is the embers cap; --resume below means re-submitting picks up
# from the latest orbax checkpoint.
#
# EXP_NAME defaults to `_v2` so the buggy `_v1` checkpoint dir (which has
# (16, 1024) per-arm shapes from the pre-fix code) is never touched. Loading
# `_v1` into the fixed model would fail the shape check anyway.

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

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_3d_gen_ego_phase2_v2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-3d-gen-ego-phase2] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-3d-gen-ego-phase2] config=pi05_mesa_bimanual_lora_3d_gen_ego_phase2 exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-3d-gen-ego-phase2] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-3d-gen-ego-phase2] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_ego_phase2/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_ego_phase2 \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-3d-gen-ego-phase2] done."
