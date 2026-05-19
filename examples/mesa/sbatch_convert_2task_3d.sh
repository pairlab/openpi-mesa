#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH -J mesa_convert_2task_3d
#SBATCH -o examples/mesa/slurm_logs/convert_%j.out
#SBATCH -e examples/mesa/slurm_logs/convert_%j.err
#
# Build the 2-task (apple_tray_on + bottled_water_tray_on, 60 demos each)
# Mesa bimanual LeRobot dataset with 3D inputs:
#   - RGB videos per cam (same as 2D variant)
#   - Metric depth per cam (linearized from MuJoCo z-buffer, float32)
#   - Per-cam intrinsics + extrinsics (float32)
#   - Per-arm hand_mat (float32)
#
# CPU-bound (h5py read + parquet + AV1 video encode); GPU is requested only
# because this queue is shared with training — the conversion itself does
# not touch CUDA.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

STAGE_DIR="/storage/scratch1/5/fchang40/mesa_bimanual_2task_3d_stage"
REPO_ID="mesa_bimanual_2task_3d"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export GIT_LFS_SKIP_SMUDGE=1
export HDF5_USE_FILE_LOCKING=FALSE

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$REPO_DIR/examples/mesa/slurm_logs"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "[convert] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[convert] stage=$STAGE_DIR"
echo "[convert] target=$HF_LEROBOT_HOME/$REPO_ID"
echo "[convert] start: $(date -Is)"

uv run examples/mesa/convert_mesa_data_to_lerobot.py \
    --raw-dir "$STAGE_DIR" \
    --repo-id "$REPO_ID" \
    --fps 20 \
    --include-3d

echo "[convert] done: $(date -Is)"
echo "[convert] output:"
ls -lh "$HF_LEROBOT_HOME/$REPO_ID/meta/" || true
du -sh "$HF_LEROBOT_HOME/$REPO_ID" 2>/dev/null || true
