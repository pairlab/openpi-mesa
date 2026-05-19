#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_offline_eval_gen_camdrop
#SBATCH -o examples/mesa/slurm_logs/offline_eval_gen_camdrop_%j.out
#SBATCH -e examples/mesa/slurm_logs/offline_eval_gen_camdrop_%j.err
#
# Offline prediction check for the camdrop-trained pi0.5 LoRA checkpoints
# (2D + 3D variants of the 4-camera MimicGen Mesa bimanual dataset). Evaluates
# action-MSE / per-dim error on training batches with the rollout-time camera
# configuration: only the egocentric view enabled, all shoulder views masked
# out via image_masks. This is the most relevant offline signal for the
# planned single-camera (egocentric) rollouts.
#
# Runs on L40S GPU (the same hardware that successfully runs the LoRA training
# job, so the JAX stack is known-good). Inference on RTX 6000 segfaults during
# XLA compile with this stack, so we route GPU evals to L40S.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   CONFIG_NAME    : TrainConfig name (e.g. pi05_mesa_bimanual_lora_3d_gen_camdrop)
#   EXP_NAME       : experiment subdir (default: ${CONFIG_NAME#pi05_}_v1)
#   CHECKPOINT_DIR : explicit step dir; if unset, picks the highest-numeric
#                    subdir under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME
#   KEEP_CAMERAS   : comma-separated mask keys to keep True (default:
#                    egocentric_0_rgb). All other cameras are masked.
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

if [[ -z "${CONFIG_NAME:-}" ]]; then
    echo "[offline-eval] error: CONFIG_NAME must be set (e.g. pi05_mesa_bimanual_lora_3d_gen_camdrop)" >&2
    exit 2
fi

EXP_NAME="${EXP_NAME:-${CONFIG_NAME#pi05_}_v1}"
KEEP_CAMERAS="${KEEP_CAMERAS:-egocentric_0_rgb}"
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
    CHECKPOINT_DIR=$(ls -d "$EXP_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$CHECKPOINT_DIR" ]]; then
        echo "[offline-eval] error: no numeric step subdir under $EXP_DIR" >&2
        exit 3
    fi
fi

echo "[offline-eval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[offline-eval] config=$CONFIG_NAME exp=$EXP_NAME"
echo "[offline-eval] checkpoint=$CHECKPOINT_DIR"
echo "[offline-eval] keep_cameras=$KEEP_CAMERAS  batch=$BATCH_SIZE  num_batches=$NUM_BATCHES"

uv run python examples/mesa/offline_eval_3d.py \
    --config-name "$CONFIG_NAME" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-sample-steps "$NUM_SAMPLE_STEPS" \
    --keep-cameras "$KEEP_CAMERAS" \
    $SHUFFLE_FLAG

echo "[offline-eval] done."
