#!/bin/bash
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0,atl1-1-01-010-35-0
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pi05_mesa_bimanual_eval_ego_ood
#SBATCH -o examples/mesa/slurm_logs/eval_ego_ood_%j.out
#SBATCH -e examples/mesa/slurm_logs/eval_ego_ood_%j.err
#
# Closed-loop sim rollout for the single-egocentric ``*_gen_ego`` LoRA
# checkpoints under an out-of-distribution camera pose distribution. Mirrors
# sbatch_eval_camdrop_ego.sh but layers a runtime override on top of the
# eval-split BDDL randomization, shifting the egocentric camera base so the
# entire sampling envelope is disjoint from what training saw.
#
# Reference: training egocentric BDDL randomization is r=1.3 ± 0.15,
# theta=0.395 ± 0.2, phi=0 ± π/6 (≈±0.524 rad). The default OOD override
# below shifts phi to π/2 with phi_radius=0.2, so sampled phi ∈ [1.37, 1.77]
# — guaranteed disjoint from training's [-0.524, +0.524] (gap ≥ 0.85 rad ≈
# 49°). Camera physically moves to the +y side of the table while still
# looking at workspace center. r/theta keep their BDDL ranges so the OOD
# axis is purely azimuth (single-axis OOD = clean signal).
#
# Runtime knobs (override via pre-export then ``sbatch --export=ALL ...``):
#   CONFIG          training config name. Default:
#                   pi05_mesa_bimanual_lora_3d_gen_ego. 2D variant:
#                   pi05_mesa_bimanual_lora_2d_gen_ego.
#   STEP            checkpoint step. Default 29999.
#   CHECKPOINT_DIR  full path to step dir; overrides STEP.
#   EXP_NAME        output tree experiment name. Default ${CONFIG#pi05_}_ego.
#   VARIANT_NAME    output tree variant name. Default
#                   ${STEP_TAG}-ego-eval-ood. Bump per (override-preset)
#                   replicate or the launcher short-circuits.
#   NUM_ROLLOUTS    rollouts per task (default 30; both tasks → 60 total).
#   NUM_WORKERS     env workers (default 4 — keep ≤ cpus-per-task).
#   REPLAN_STEPS    actions executed before replanning (default 5).
#   EVAL_SET_NAME   default mesa_bimanual.
#   EVAL_SPLIT      default eval. NB: ``train`` is broken (see
#                   markdowns/camdrop_2d_vs_3d_eval.md §4).
#   TASK_FILTER     space-separated. Default "apple_tray_on lime_bowl_on".
#   CAMERA_OVERRIDES  JSON string passed verbatim to launch_eval.py
#                   --camera-overrides. Default shifts egocentric base to
#                   phi=π/2 (≈1.5708) with phi_radius=0.2 — every sampled
#                   pose is ≥0.85 rad past training's max |phi|. Empty
#                   string disables (in-distribution eval).
#   SEED            integer env-init seed. Default 7. Bump + new VARIANT_NAME
#                   for replicates.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

CONFIG="${CONFIG:-pi05_mesa_bimanual_lora_3d_gen_ego}"
STEP="${STEP:-29999}"
EXP_NAME_DEFAULT="${CONFIG#pi05_}_ego"
EXP_NAME="${EXP_NAME:-$EXP_NAME_DEFAULT}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
EVAL_SET_NAME="${EVAL_SET_NAME:-mesa_bimanual}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
TASK_FILTER="${TASK_FILTER:-apple_tray_on lime_bowl_on}"
SEED="${SEED:-7}"

# Default OOD egocentric override: shift azimuth base so the entire sample
# envelope lies outside training's reachable phi. Training: phi ∈ ±π/6 ≈
# ±0.524. Override: phi=π/2 ± 0.2 = [1.37, 1.77] — disjoint with margin
# ≥0.85 rad. Only phi/phi_radius are touched; r and theta keep their BDDL
# ranges (in-distribution) so the OOD axis is pure azimuth.
DEFAULT_CAMERA_OVERRIDES='{"egocentric":{"phi":1.5707963267948966,"phi_radius":0.2}}'
CAMERA_OVERRIDES="${CAMERA_OVERRIDES-$DEFAULT_CAMERA_OVERRIDES}"

CHECKPOINT_BASE_DIR="$PROJECT_DIR/checkpoints"
EXP_DIR="$CHECKPOINT_BASE_DIR/$CONFIG/${CONFIG#pi05_}_v1"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$EXP_DIR/$STEP}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[eval-ego-ood] checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 2
fi
STEP_TAG="$(basename "$CHECKPOINT_DIR")"
VARIANT_NAME="${VARIANT_NAME:-${STEP_TAG}-ego-${EVAL_SPLIT}-ood}"

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

# Single-cam ego configs declare cameras=("egocentric",) — bench must send
# only that. Override CAMERA_NAMES in the env if the config differs.
if [ -n "${CAMERA_NAMES:-}" ]; then
    read -r -a CAMERAS <<< "$CAMERA_NAMES"
else
    CAMERAS=(egocentric)
fi

echo "[eval-ego-ood] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[eval-ego-ood] config=$CONFIG step=$STEP_TAG"
echo "[eval-ego-ood] checkpoint=$CHECKPOINT_DIR"
echo "[eval-ego-ood] exp=$EXP_NAME variant=$VARIANT_NAME"
echo "[eval-ego-ood] eval_set=$EVAL_SET_NAME split=$EVAL_SPLIT tasks=$TASK_FILTER"
echo "[eval-ego-ood] rollouts/task=$NUM_ROLLOUTS workers=$NUM_WORKERS"
echo "[eval-ego-ood] cameras=${CAMERAS[*]}"
echo "[eval-ego-ood] seed=$SEED"
echo "[eval-ego-ood] camera_overrides=$CAMERA_OVERRIDES"

CAMERA_OVERRIDES_ARG=()
if [ -n "$CAMERA_OVERRIDES" ]; then
    CAMERA_OVERRIDES_ARG=(--camera-overrides "$CAMERA_OVERRIDES")
fi

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
    "${CAMERA_OVERRIDES_ARG[@]}" \
    --video-out-path "$REPO_DIR/experiments/vla_benchmark"

SUMMARY="$REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/statistics/final_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "[eval-ego-ood] final_summary:"
    cat "$SUMMARY"
fi
echo "[eval-ego-ood] videos: $REPO_DIR/experiments/vla_benchmark/$EVAL_SET_NAME/$EXP_NAME/$VARIANT_NAME/videos"
echo "[eval-ego-ood] done."
