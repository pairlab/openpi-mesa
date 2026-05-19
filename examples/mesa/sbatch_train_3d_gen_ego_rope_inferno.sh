#!/bin/bash
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --nodelist=atl1-1-01-010-29-0,atl1-1-01-010-31-0,atl1-1-01-010-33-0,atl1-1-01-010-35-0,atl1-1-03-004-29-0
# NOTE: --exclude was silently overridden by a PACE site policy (job 7830146
# landed on atl1-1-03-007-31-0 despite the exclude and silently exited at
# 2:57 walltime — the known bad-node failure mode). Switched to a positive
# --nodelist with only the verified-good ideas L40S nodes.
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_ego_rope_train_inferno
#SBATCH -o examples/mesa/slurm_logs/train_3d_gen_ego_rope_inferno_%j.out
#SBATCH -e examples/mesa/slurm_logs/train_3d_gen_ego_rope_inferno_%j.err
#
# RoPE-arch variant of sbatch_train_3d_gen_ego_inferno.sh — same TrainConfig
# (pi05_mesa_bimanual_lora_3d_gen_ego), same dataset, same batch size, same
# num_train_steps. The TrainConfig now instantiates Pi0Adapt3RConfig with the
# new 3D-RoPE-on-image-tokens path (see markdowns/A.md). Old-arch checkpoints
# under mesa_bimanual_lora_3d_gen_ego_v1 are NOT loadable into this structure,
# so this script uses a fresh EXP_NAME and starts from pi05_base weights.
# --resume is still safe to pass: it kicks in once this run writes its first
# checkpoint, so resubmissions of this same script continue the run.

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

EXP_NAME="${EXP_NAME:-mesa_bimanual_lora_3d_gen_ego_rope_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openpi_mesa}"
CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "[train-3d-gen-ego-rope-inferno] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train-3d-gen-ego-rope-inferno] config=pi05_mesa_bimanual_lora_3d_gen_ego exp=$EXP_NAME batch=$BATCH_SIZE"
echo "[train-3d-gen-ego-rope-inferno] wandb: $WANDB_ENTITY/$WANDB_PROJECT_NAME/$EXP_NAME"
echo "[train-3d-gen-ego-rope-inferno] checkpoints -> $CHECKPOINT_BASE_DIR/pi05_mesa_bimanual_lora_3d_gen_ego/$EXP_NAME"

uv run scripts/train.py pi05_mesa_bimanual_lora_3d_gen_ego \
    --exp-name="$EXP_NAME" \
    --project-name="$WANDB_PROJECT_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --batch-size="$BATCH_SIZE" \
    --resume

echo "[train-3d-gen-ego-rope-inferno] done."
