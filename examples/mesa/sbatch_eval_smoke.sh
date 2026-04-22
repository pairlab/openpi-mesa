#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem-per-gpu=120G
#SBATCH -t 2:00:00
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_eval_smoke
#SBATCH -o examples/mesa/slurm_logs/eval_%j.out
#SBATCH -e examples/mesa/slurm_logs/eval_%j.err
#
# Overfit eval for the pi05_mesa_bimanual_lora checkpoint: runs the trained
# model against the four init-states that were also in its training set
# (mesa_bimanual/apple_tray_on, overfit split).
#
# Env overrides:
#   CHECKPOINT_DIR   step dir to load (default: latest under v2 exp)
#   EXP_NAME         experiment name in the output tree (default: pi05_mesa_smoke)
#   VARIANT_NAME     variant name (default: ${step}-smoke)
#   NUM_ROLLOUTS     rollouts per task (default: 4)
#   NUM_WORKERS      env workers (default: 2)
#   REPLAN_STEPS     actions executed before replanning (default: 5)
#   EVAL_SET_NAME    default: mesa_bimanual
#   EVAL_SPLIT       default: overfit
#   TASK_FILTER      default: apple_tray_on

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/pi05_mesa_bimanual_lora/pi05_bimanual_2task_lora_v2/9000}"
STEP_TAG="$(basename "$CHECKPOINT_DIR")"

EXP_NAME="${EXP_NAME:-pi05_mesa_smoke}"
VARIANT_NAME="${VARIANT_NAME:-${STEP_TAG}-smoke}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
EVAL_SET_NAME="${EVAL_SET_NAME:-mesa_bimanual}"
EVAL_SPLIT="${EVAL_SPLIT:-overfit}"
TASK_FILTER="${TASK_FILTER:-apple_tray_on}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 2
fi

# Env (mirrors sbatch_train.sh: everything off-home to respect 20G quota).
export HF_LEROBOT_HOME="/storage/project/r-agarg35-0"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache"
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Turing (CC 7.5, e.g. Quadro RTX 6000) flakes with recent JAX bf16 kernels.
# Disable cuDNN fused-attn and Triton GEMM fallbacks; harmless on Ampere+.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_cudnn_fmha=false --xla_gpu_enable_triton_gemm=false"
export HDF5_USE_FILE_LOCKING=FALSE

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR" \
         "$REPO_DIR/examples/mesa/slurm_logs" "$REPO_DIR/experiments/vla_benchmark"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "[eval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[eval] checkpoint=$CHECKPOINT_DIR"
echo "[eval] out=$REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME"

uv run examples/mesa/launch_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --exp-name "$EXP_NAME" \
    --variant-name "$VARIANT_NAME" \
    --eval-set-name "$EVAL_SET_NAME" \
    --eval-split "$EVAL_SPLIT" \
    --task-filter $TASK_FILTER \
    --num-rollouts-per-task "$NUM_ROLLOUTS" \
    --num-env-workers "$NUM_WORKERS" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-path "$REPO_DIR/experiments/vla_benchmark"

SUMMARY="$REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/statistics/final_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "[eval] final_summary:"
    cat "$SUMMARY"
fi
echo "[eval] videos: $REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/videos"
echo "[eval] done."
