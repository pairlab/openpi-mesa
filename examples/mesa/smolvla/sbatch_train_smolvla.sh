#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J smolvla_mesa_bimanual_train
#SBATCH -o examples/mesa/slurm_logs/smolvla_train_%j.out
#SBATCH -e examples/mesa/slurm_logs/smolvla_train_%j.err

set -euo pipefail

REPO_DIR="/storage/project/r-agarg35-0/nnguyen349/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/nnguyen349"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export HF_HOME="$PROJECT_DIR/hf_cache"
export TMPDIR="$PROJECT_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export GIT_LFS_SKIP_SMUDGE=1

EXP_NAME="${EXP_NAME:-smolvla_bimanual_2d}"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/smolvla_bimanual/$EXP_NAME"
mkdir -p "$OUTPUT_DIR" "$REPO_DIR/examples/mesa/slurm_logs"

echo "[train] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train] exp=$EXP_NAME output=$OUTPUT_DIR"

uv run lerobot-train \
    --policy.type=smolvla \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --dataset.repo_id=mesa_bimanual_2task_3d \
    --batch_size=32 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable=true \
    --wandb.project=smolvla_mesa_bimanual \
    --wandb.entity=pair-diffusion \
    --resume

echo "[train] done. checkpoints at $OUTPUT_DIR"
