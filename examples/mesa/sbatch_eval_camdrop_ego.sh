#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0,atl1-1-01-010-35-0
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_eval_camdrop_ego
#SBATCH -o examples/mesa/slurm_logs/eval_camdrop_ego_%j.out
#SBATCH -e examples/mesa/slurm_logs/eval_camdrop_ego_%j.err
#
# Closed-loop sim rollout for the 4-cam ``*_camdrop`` LoRA checkpoints with all
# shoulder cameras ``image_mask``-zeroed at inference (mirrors camdrop training
# semantics — image data is still on the wire, only the egocentric view
# attends). Runs against the vla-benchmark ``train`` split so we are testing
# whether the 200-demo training set is actually overfit, not generalization.
#
# Runtime knobs (override via ``sbatch --export=ALL,VAR=value ...``):
#   CONFIG          training config name. Default:
#                   pi05_mesa_bimanual_lora_3d_gen_camdrop. 2D variant:
#                   pi05_mesa_bimanual_lora_2d_gen_camdrop.
#   STEP            checkpoint step (default 10000); resolved against
#                   $CHECKPOINT_BASE_DIR/$CONFIG/${CONFIG#pi05_}_v1/$STEP.
#   CHECKPOINT_DIR  full path to step dir; overrides STEP.
#   EXP_NAME        output tree experiment name. Default:
#                   ${CONFIG#pi05_}_ego.
#   VARIANT_NAME    output tree variant name. Default: ${STEP}-ego.
#   NUM_ROLLOUTS    rollouts per task (default: 30; both tasks ⇒ 60 total).
#   NUM_WORKERS     env workers (default: 4 — keep ≤ cpus-per-task).
#   REPLAN_STEPS    actions executed before replanning (default: 5).
#   EVAL_SET_NAME   default: mesa_bimanual.
#   EVAL_SPLIT      default: train.
#   TASK_FILTER     space-separated. Default: "apple_tray_on lime_bowl_on".
#   KEEP_CAMERAS    comma-separated bare cam names left unmasked. Default:
#                   egocentric. Pass "ALL" or "" to disable masking.
#   CAMERA_NAMES    space-separated bare cam names sent on the wire by the
#                   bench. Default: "egocentric leftshoulder rightshoulder
#                   midshoulder" (camdrop superset). Single-cam ego configs
#                   must override to "egocentric" — the policy server reads
#                   train_config.data.cameras and KeyErrors on a mismatch.
#   SEED            integer env-init seed handed to the bench. Default 7
#                   (matches launch_eval.py default). Bump per replicate
#                   when running multi-seed comparisons; pair with a unique
#                   VARIANT_NAME so the launcher does not short-circuit.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

CONFIG="${CONFIG:-pi05_mesa_bimanual_lora_3d_gen_camdrop}"
STEP="${STEP:-10000}"
EXP_NAME_DEFAULT="${CONFIG#pi05_}_ego"
EXP_NAME="${EXP_NAME:-$EXP_NAME_DEFAULT}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
EVAL_SET_NAME="${EVAL_SET_NAME:-mesa_bimanual}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
TASK_FILTER="${TASK_FILTER:-apple_tray_on lime_bowl_on}"
KEEP_CAMERAS="${KEEP_CAMERAS-egocentric}"
SEED="${SEED:-7}"
if [[ "${KEEP_CAMERAS^^}" == "ALL" ]]; then
    KEEP_CAMERAS=""
fi

CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
EXP_DIR="$CHECKPOINT_BASE_DIR/$CONFIG/${CONFIG#pi05_}_v1"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$EXP_DIR/$STEP}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[eval-camdrop-ego] checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 2
fi
STEP_TAG="$(basename "$CHECKPOINT_DIR")"
VARIANT_NAME="${VARIANT_NAME:-${STEP_TAG}-ego}"

# Env (mirrors training scripts: everything off-home to respect 20G quota).
export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_DIR/jax_cache_l40s}"
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HDF5_USE_FILE_LOCKING=FALSE

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR" \
         "$REPO_DIR/examples/mesa/slurm_logs" "$REPO_DIR/experiments/vla_benchmark"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

if [ -n "${CAMERA_NAMES:-}" ]; then
    read -r -a CAMERAS <<< "$CAMERA_NAMES"
else
    CAMERAS=(egocentric leftshoulder rightshoulder midshoulder)
fi

echo "[eval-camdrop-ego] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[eval-camdrop-ego] config=$CONFIG step=$STEP_TAG"
echo "[eval-camdrop-ego] checkpoint=$CHECKPOINT_DIR"
echo "[eval-camdrop-ego] exp=$EXP_NAME variant=$VARIANT_NAME"
echo "[eval-camdrop-ego] eval_set=$EVAL_SET_NAME split=$EVAL_SPLIT tasks=$TASK_FILTER"
echo "[eval-camdrop-ego] rollouts/task=$NUM_ROLLOUTS workers=$NUM_WORKERS"
echo "[eval-camdrop-ego] cameras=${CAMERAS[*]}  keep_cameras='${KEEP_CAMERAS}'"
echo "[eval-camdrop-ego] seed=$SEED"

uv run examples/mesa/launch_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --variant-name "$VARIANT_NAME" \
    --eval-set-name "$EVAL_SET_NAME" \
    --eval-split "$EVAL_SPLIT" \
    --task-filter $TASK_FILTER \
    --num-rollouts-per-task "$NUM_ROLLOUTS" \
    --num-env-workers "$NUM_WORKERS" \
    --replan-steps "$REPLAN_STEPS" \
    --seed "$SEED" \
    --camera-names "${CAMERAS[@]}" \
    --keep-cameras "$KEEP_CAMERAS" \
    --video-out-path "$REPO_DIR/experiments/vla_benchmark"

SUMMARY="$REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/statistics/final_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "[eval-camdrop-ego] final_summary:"
    cat "$SUMMARY"
fi
echo "[eval-camdrop-ego] videos: $REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/videos"
echo "[eval-camdrop-ego] done."
