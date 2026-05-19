#!/bin/bash
#
# Submit closed-loop rollouts for the new 3D RoPE ego LoRA
# (pi05_mesa_bimanual_lora_3d_gen_ego_rope_v1). Two checkpoint steps
# (default 15000, 18000) x {in-dist, OOD phi=pi/2 +- 0.2} for one seed.
# Single-seed snapshot — meant to be paired against the existing
# pooled 29999-step results in
# markdowns/ego_ood_camera_3d_vs_2d.md (kept under exp tree
# mesa_bimanual_lora_3d_gen_ego_ego). The new rope results land in a
# distinct exp tree (mesa_bimanual_lora_3d_gen_ego_rope_ego) so the old
# additive-PE numbers are untouched.
#
# Submitted on embers via sbatch CLI overrides; the underlying
# sbatch script's inferno header is overridden per-submission rather
# than edited in-place, so other launchers that submit it on inferno
# keep working. GPU type is selectable so we can chase whichever
# embers pool is shorter at submission time; the JAX compile cache
# is partitioned per GPU type because cross-GPU caches silently
# SIGSEGV at first JIT (memory: feedback_jax_cache_per_gpu).
#
# Knobs:
#   QUEUE    embers | inferno. Default embers. inferno is l40s-only here
#            (uses the script's hardcoded inferno header, no CLI override).
#   GPU      l40s | h100. Default l40s. Ignored when QUEUE=inferno
#            (always l40s).
#   STEPS    space-separated checkpoint steps. Default "15000 18000".
#   SEED     env-init seed. Default 7.
#   DRY_RUN  print only.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
SBATCH_SCRIPT="$REPO_DIR/examples/mesa/sbatch_eval_ego_ood.sh"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "missing sbatch script: $SBATCH_SCRIPT" >&2
    exit 2
fi

CONFIG="pi05_mesa_bimanual_lora_3d_gen_ego"
ROPE_EXP_DIR="$PROJECT_DIR/checkpoints/$CONFIG/mesa_bimanual_lora_3d_gen_ego_rope_v1"
EXP_NAME_VAL="mesa_bimanual_lora_3d_gen_ego_rope_ego"
SEED="${SEED:-7}"
read -r -a STEPS_ARR <<< "${STEPS:-15000 18000}"

PHI_PI_OVER_2="1.5707963267948966"
PHI90_OVERRIDES="{\"egocentric\":{\"phi\":${PHI_PI_OVER_2},\"phi_radius\":0.2}}"

# Queue-dependent SBATCH overrides. Embers walltime is hard-capped at
# 8h (memory: feedback_embers_walltime) regardless of partition
# TIMELIMIT; inferno keeps the script's 12h header. JAX cache is
# partitioned per GPU type since cross-GPU caches silently SIGSEGV at
# first JIT (memory: feedback_jax_cache_per_gpu).
QUEUE="${QUEUE:-embers}"
GPU="${GPU:-l40s}"
case "$QUEUE" in
    embers)
        case "$GPU" in
            l40s)
                SBATCH_OVERRIDES=(
                    -A gts-agarg35 -q embers -p gpu-l40s
                    --gres=gpu:l40s:1 -t 8:00:00
                )
                export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache_l40s"
                ;;
            h100)
                SBATCH_OVERRIDES=(
                    -A gts-agarg35 -q embers -p gpu-h100
                    --gres=gpu:h100:1 -t 8:00:00
                )
                export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache_h100"
                ;;
            *)
                echo "GPU must be l40s or h100 (got: $GPU)" >&2
                exit 2
                ;;
        esac
        ;;
    inferno)
        # Use the sbatch script's hardcoded inferno header (account
        # gts-agarg35-ideas_l40s, queue inferno, partition gpu-l40s,
        # gres gpu:l40s:1, walltime 12h). No CLI overrides — keeps the
        # behavior identical to the existing inferno-targeting launchers.
        SBATCH_OVERRIDES=()
        export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/jax_cache_l40s"
        ;;
    *)
        echo "QUEUE must be embers or inferno (got: $QUEUE)" >&2
        exit 2
        ;;
esac

RECORD_FILE="$REPO_DIR/markdowns/rope_ego_eval_jobs.tsv"
mkdir -p "$(dirname "$RECORD_FILE")"
if [ "$DRY_RUN" != "1" ] && [ ! -f "$RECORD_FILE" ]; then
    printf 'timestamp\tjobid\tstep\tcondition\tseed\tvariant\n' > "$RECORD_FILE"
fi

submitted=0
skipped=0

submit_one() {
    local step_tag="$1"
    local condition="$2"   # "indist" or "phi90"
    local seed="$3"
    local overrides="$4"   # "" for in-dist
    local checkpoint_dir="$ROPE_EXP_DIR/$step_tag"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "[skip] missing checkpoint: $checkpoint_dir" >&2
        skipped=$((skipped + 1))
        return 0
    fi

    local variant
    if [ "$condition" = "indist" ]; then
        variant="${step_tag}-ego-eval-seed${seed}"
    else
        variant="${step_tag}-ego-eval-ood-${condition}-seed${seed}"
    fi
    local summary="$REPO_DIR/experiments/vla_benchmark/mesa_bimanual/${EXP_NAME_VAL}/${variant}/statistics/final_summary.json"
    if [ -f "$summary" ]; then
        echo "[skip] already done: $EXP_NAME_VAL/$variant"
        skipped=$((skipped + 1))
        return 0
    fi

    export CONFIG
    export CHECKPOINT_DIR="$checkpoint_dir"
    export EXP_NAME="$EXP_NAME_VAL"
    export VARIANT_NAME="$variant"
    export CAMERA_OVERRIDES="$overrides"
    export SEED="$seed"
    # Eval script defaults STEP -> "29999" if unset, but the path is
    # taken from CHECKPOINT_DIR which we override explicitly.
    unset STEP || true

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] step=$step_tag condition=$condition seed=$seed variant=$variant"
        return 0
    fi

    local out jobid ts
    out=$(sbatch "${SBATCH_OVERRIDES[@]}" --export=ALL "$SBATCH_SCRIPT")
    jobid=$(echo "$out" | awk '{print $NF}')
    echo "[submit] jobid=$jobid step=$step_tag condition=$condition seed=$seed variant=$variant"
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ts" "$jobid" "$step_tag" "$condition" "$seed" "$variant" \
        >> "$RECORD_FILE"
    submitted=$((submitted + 1))
}

for step_tag in "${STEPS_ARR[@]}"; do
    submit_one "$step_tag" "indist" "$SEED" ""
    submit_one "$step_tag" "phi90"  "$SEED" "$PHI90_OVERRIDES"
done

echo "[summary] submitted=$submitted skipped=$skipped record=$RECORD_FILE"
