#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0,atl1-1-01-010-35-0
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH -t 0:15:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH -J render_ood_ego_check
#SBATCH -o examples/mesa/slurm_logs/render_ood_ego_check_%j.out
#SBATCH -e examples/mesa/slurm_logs/render_ood_ego_check_%j.err
#
# 1-shot offscreen render comparing in-distribution vs OOD egocentric framing
# for the same scene. Output: ood_ego_check.png in repo root.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$REPO_DIR/examples/mesa/slurm_logs"
export TMPDIR="$SCRATCH_BASE/tmp"

OUTPUT="${OUTPUT:-$REPO_DIR/ood_ego_check.png}"
TASK="${TASK:-apple_tray_on}"
INSTANCE="${INSTANCE:-0}"
CAMERA_OVERRIDES="${CAMERA_OVERRIDES-{\"egocentric\":{\"phi\":1.5707963267948966,\"phi_radius\":0.2}}}"

VLA_BENCHMARK_PYTHON="/storage/project/r-agarg35-0/fchang40/venvs/vla-benchmark/bin/python"

echo "[render] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[render] task=$TASK instance=$INSTANCE"
echo "[render] camera_overrides=$CAMERA_OVERRIDES"
echo "[render] output=$OUTPUT"

"$VLA_BENCHMARK_PYTHON" examples/mesa/render_ood_ego_check.py \
    --task "$TASK" \
    --instance "$INSTANCE" \
    --camera-overrides "$CAMERA_OVERRIDES" \
    --output "$OUTPUT"

echo "[render] done."
