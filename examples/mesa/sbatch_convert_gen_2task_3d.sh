#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH -J mesa_convert_gen_2task_3d
#SBATCH -o examples/mesa/slurm_logs/convert_gen_%j.out
#SBATCH -e examples/mesa/slurm_logs/convert_gen_%j.err
#
# Build the MimicGen-generated 2-task (apple_tray_on + lime_bowl_on, 200 demos
# each) Mesa bimanual LeRobot dataset with 3D inputs:
#   - RGB videos per cam (4 cams: egocentric + {left,mid,right}shoulder)
#   - Metric depth per cam (linearized from MuJoCo z-buffer, float32)
#   - Per-cam intrinsics + extrinsics (float32)
#   - Per-arm hand_mat (float32)
#
# The source tree uses the per-demo layout
# (``<task>/demo/tmp/*.hdf5``, one demo per file) which the converter now
# walks directly — no staging needed. CPU-bound (h5py read + parquet + AV1
# video encode); the GPU is requested only because the rtx6000 embers queue
# is shared with training.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

RAW_DIR="/storage/project/r-agarg35-0/shared/vla_benchmark_data/apr_23/gen_data"
REPO_ID="mesa_bimanual_gen_2task_3d"

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
echo "[convert] raw=$RAW_DIR"
echo "[convert] target=$HF_LEROBOT_HOME/$REPO_ID"
echo "[convert] start: $(date -Is)"

uv run examples/mesa/convert_mesa_data_to_lerobot.py \
    --raw-dir "$RAW_DIR" \
    --repo-id "$REPO_ID" \
    --fps 20 \
    --include-3d \
    --cameras egocentric leftshoulder rightshoulder midshoulder

echo "[convert] done: $(date -Is)"
echo "[convert] output:"
ls -lh "$HF_LEROBOT_HOME/$REPO_ID/meta/" || true
