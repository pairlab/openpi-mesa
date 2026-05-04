#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --mem-per-gpu=120G
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH -J smolvla_mesa_bimanual_eval
#SBATCH -o examples/mesa/slurm_logs/smolvla_eval_%j.out
#SBATCH -e examples/mesa/slurm_logs/smolvla_eval_%j.err
#
# Env overrides:
#   CHECKPOINT_DIR   pretrained_model dir to load (required)
#   EXP_NAME         experiment name in output tree (default: smolvla_mesa_eval)
#   VARIANT_NAME     variant tag (default: <step>-eval)
#   NUM_ROLLOUTS     rollouts per task (default: 4)
#   NUM_WORKERS      env workers (default: 2)
#   REPLAN_STEPS     actions executed before replanning (default: 5)
#   EVAL_SPLIT       overfit | train | ... (default: overfit)
#   TASK_FILTER      space-separated task names (default: apple_tray_on)

set -euo pipefail

REPO_DIR="/storage/project/r-agarg35-0/nnguyen349/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/nnguyen349"
cd "$REPO_DIR"

if [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "ERROR: CHECKPOINT_DIR must be set to the pretrained_model dir" >&2
    exit 2
fi
STEP_TAG="$(basename "$(dirname "$CHECKPOINT_DIR")")"

EXP_NAME="${EXP_NAME:-smolvla_mesa_eval}"
VARIANT_NAME="${VARIANT_NAME:-${STEP_TAG}-eval}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
EVAL_SPLIT="${EVAL_SPLIT:-overfit}"
TASK_FILTER="${TASK_FILTER:-apple_tray_on}"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export HF_HOME="$PROJECT_DIR/hf_cache"
export TMPDIR="$PROJECT_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export GIT_LFS_SKIP_SMUDGE=1
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$REPO_DIR/examples/mesa/slurm_logs" "$REPO_DIR/experiments/vla_benchmark"

echo "[eval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[eval] checkpoint=$CHECKPOINT_DIR"
echo "[eval] exp=$EXP_NAME variant=$VARIANT_NAME"

uv run examples/mesa/launch_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --server-script "examples/mesa/smolvla/smolvla_policy_server.py" \
    --exp-name "$EXP_NAME" \
    --variant-name "$VARIANT_NAME" \
    --eval-set-name "mesa_bimanual" \
    --eval-split "$EVAL_SPLIT" \
    --task-filter $TASK_FILTER \
    --num-rollouts-per-task "$NUM_ROLLOUTS" \
    --num-env-workers "$NUM_WORKERS" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-path "$REPO_DIR/experiments/vla_benchmark"

SUMMARY="$REPO_DIR/experiments/vla_benchmark/mesa_bimanual/$EXP_NAME/$VARIANT_NAME/statistics/final_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "[eval] final_summary:"
    cat "$SUMMARY"
fi
echo "[eval] done."
