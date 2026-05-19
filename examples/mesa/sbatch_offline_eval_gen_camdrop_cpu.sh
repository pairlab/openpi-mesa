#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-rtx6000
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J pi05_mesa_bimanual_offline_eval_gen_camdrop_cpu
#SBATCH -o examples/mesa/slurm_logs/offline_eval_gen_camdrop_cpu_%j.out
#SBATCH -e examples/mesa/slurm_logs/offline_eval_gen_camdrop_cpu_%j.err
#
# CPU-only offline prediction check for the camdrop-trained pi0.5 LoRA
# checkpoints (2D + 3D variants of the 4-camera MimicGen Mesa bimanual
# dataset). Forces JAX onto CPU and runs on a gpu-rtx6000/embers node purely
# for the larger CPU/RAM allotment per node — RTX 6000 GPU inference segfaults
# during XLA compile with this stack, so the GPU is left untouched.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   CONFIG_NAME    : TrainConfig name (e.g. pi05_mesa_bimanual_lora_3d_gen_camdrop)
#   EXP_NAME       : experiment subdir (default: ${CONFIG_NAME#pi05_}_v1)
#   STEP           : checkpoint step (e.g. 7000); ignored if CHECKPOINT_DIR is set
#   CHECKPOINT_DIR : explicit step dir; overrides STEP
#   KEEP_CAMERAS   : comma-separated mask keys to keep True (default:
#                    egocentric_0_rgb). Pass "ALL" or "" to disable masking
#                    and use every camera.
#   BATCH_SIZE     : default 4
#   NUM_BATCHES    : default 3
#   NUM_SAMPLE_STEPS : default 10
#   SHUFFLE        : "1" to draw samples across all episodes (default 0)

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache"
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

if [[ -z "${CONFIG_NAME:-}" ]]; then
    echo "[offline-eval-cpu] error: CONFIG_NAME must be set (e.g. pi05_mesa_bimanual_lora_3d_gen_camdrop)" >&2
    exit 2
fi

EXP_NAME="${EXP_NAME:-${CONFIG_NAME#pi05_}_v1}"
KEEP_CAMERAS="${KEEP_CAMERAS-egocentric_0_rgb}"
if [[ "${KEEP_CAMERAS^^}" == "ALL" ]]; then
    KEEP_CAMERAS=""
fi
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_BATCHES="${NUM_BATCHES:-3}"
NUM_SAMPLE_STEPS="${NUM_SAMPLE_STEPS:-10}"
SHUFFLE="${SHUFFLE:-0}"
SHUFFLE_FLAG=""
if [[ "$SHUFFLE" == "1" ]]; then
    SHUFFLE_FLAG="--shuffle"
fi

CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
EXP_DIR="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    if [[ -n "${STEP:-}" ]]; then
        CHECKPOINT_DIR="$EXP_DIR/$STEP"
    else
        CHECKPOINT_DIR=$(ls -d "$EXP_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
    fi
fi

if [[ -z "$CHECKPOINT_DIR" || ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[offline-eval-cpu] error: checkpoint dir not found: ${CHECKPOINT_DIR:-<empty>}" >&2
    exit 3
fi

echo "[offline-eval-cpu] node=$(hostname) cpu=$(nproc) platform=cpu-only"
echo "[offline-eval-cpu] config=$CONFIG_NAME exp=$EXP_NAME"
echo "[offline-eval-cpu] checkpoint=$CHECKPOINT_DIR"
echo "[offline-eval-cpu] keep_cameras='${KEEP_CAMERAS}'  batch=$BATCH_SIZE  num_batches=$NUM_BATCHES"

uv run python examples/mesa/offline_eval_3d.py \
    --config-name "$CONFIG_NAME" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-sample-steps "$NUM_SAMPLE_STEPS" \
    --keep-cameras "$KEEP_CAMERAS" \
    $SHUFFLE_FLAG

echo "[offline-eval-cpu] done."
