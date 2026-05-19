#!/bin/bash
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --nodelist=atl1-1-01-010-29-0,atl1-1-01-010-31-0,atl1-1-01-010-33-0,atl1-1-01-010-35-0,atl1-1-03-004-29-0
# NOTE: --exclude= is silently overridden by a PACE site policy on the inferno
# queue (see markdowns/B.md and the bad-node memory). Use a positive
# --nodelist= of verified-good ideas L40S nodes instead.
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_ego_phase2_train_inferno
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_ego_phase2_inferno_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_ego_phase2_inferno_%j.err
#
# Phase 2 of the 3D RoPE work — per-arm action-token split with 3D RoPE keyed
# by per-arm gripper xyz (markdowns/B.md §8). Same dataset and budget as the
# Phase 1 run (`pi05_mesa_bimanual_lora_3d_gen_ego_rope_v1`); the TrainConfig
# `pi05_mesa_bimanual_lora_3d_gen_ego_phase2` instantiates
# `Pi0Adapt3RBimanualConfig` and threads `extra_arms=(1,)` through the data
# pipeline so robot1's hand_mat is available alongside the reference-arm
# pose. Phase 2 introduces three fresh-init params
# (`action_in_proj_per_arm`, `action_out_proj_per_arm`, `temporal_pos_embed`);
# the train config carves them out via `extra_missing_regex`. Existing
# Phase 1 checkpoints are NOT loadable into the Phase 2 structure — this
# run starts fresh from `pi05_base`.

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

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_3d_gen_ego_phase2_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-3d-gen-ego-phase2-inferno] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-3d-gen-ego-phase2-inferno] config=pi05_mesa_bimanual_lora_3d_gen_ego_phase2 exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-3d-gen-ego-phase2-inferno] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-3d-gen-ego-phase2-inferno] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_ego_phase2/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_ego_phase2 \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-3d-gen-ego-phase2-inferno] done."
