#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_lora_normstats
#SBATCH -o examples/mesa/slurm_logs/normstats_%j.out
#SBATCH -e examples/mesa/slurm_logs/normstats_%j.err
#
# Compute normalization statistics for TrainConfig `pi05_mesa_bimanual_lora`.
# Reads the LeRobot dataset at
#   $HF_LEROBOT_HOME/fchang40/mesa_bimanual_2task
# and writes per-feature norm stats to
#   ./assets/pi05_mesa_bimanual_lora/fchang40/mesa_bimanual_2task/
# Must be run before `scripts/train.py` — the data loader errors without it.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0"
export UV_PROJECT_ENVIRONMENT="/storage/project/r-agarg35-0/fchang40/venvs/openpi-mesa"
export UV_CACHE_DIR="/storage/project/r-agarg35-0/fchang40/uv_cache"
export GIT_LFS_SKIP_SMUDGE=1

# Keep /tmp usage off the home quota (20G).
export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "[normstats] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[normstats] config=pi05_mesa_bimanual_lora repo_id=fchang40/mesa_bimanual_2task"

uv run scripts/compute_norm_stats.py --config-name pi05_mesa_bimanual_lora

echo "[normstats] done. Wrote:"
ls -la "$REPO_DIR/assets/pi05_mesa_bimanual_lora/fchang40/mesa_bimanual_2task/" || true
