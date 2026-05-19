#!/bin/bash
#
# Submit the missing seed × azimuth-severity matrix for the egocentric OOD
# camera study (markdowns/ego_ood_camera_3d_vs_2d.md, "Next steps" §1).
#
# Existing artifacts cover seed=7 OOD at phi49/60/90/120 for both the 2D and
# 3D ego LoRAs at step 29999, plus in-distribution baselines at seeds
# 7/11/23. This script enqueues the seed=11 and seed=23 OOD runs across all
# four severities for both architectures (2 archs * 4 phi * 2 seeds = 16
# jobs) so the pooled OOD penalty SE drops from ~4.5 pp -> ~2.5 pp.
#
# Each child job is a vanilla submission of sbatch_eval_ego_ood.sh; we only
# vary CONFIG, SEED, VARIANT_NAME and CAMERA_OVERRIDES via env vars. We
# pre-export those and then submit with plain --export=ALL so the
# comma-bearing CAMERA_OVERRIDES JSON is not parsed by Slurm's --export
# splitter (memory: feedback_sbatch_export_commas).
#
# Variant naming extends the existing seed-7 dirs:
#   29999-ego-eval-ood-phi{49,60,90,120}-seed{11,23}
# launch_eval.py short-circuits on existing final_summary.json, so re-runs
# are no-ops and naming collisions are detected up front.
#
# Usage:
#   bash examples/mesa/submit_ego_ood_multiseed.sh           # submit all 16
#   DRY_RUN=1 bash examples/mesa/submit_ego_ood_multiseed.sh # print only

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
SBATCH_SCRIPT="$REPO_DIR/examples/mesa/sbatch_eval_ego_ood.sh"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "missing sbatch script: $SBATCH_SCRIPT" >&2
    exit 2
fi

CONFIGS=(
    "pi05_mesa_bimanual_lora_2d_gen_ego"
    "pi05_mesa_bimanual_lora_3d_gen_ego"
)

# (label, phi base value as a python-printable float string)
PHI_LABELS=(phi49 phi60 phi90 phi120)
PHI_VALUES=(
    "0.85"
    "1.0471975511965976"   # pi/3
    "1.5707963267948966"   # pi/2
    "2.0943951023931953"   # 2*pi/3
)

SEEDS=(11 23)
STEP_TAG="29999"
RECORD_FILE="$REPO_DIR/markdowns/ego_ood_multiseed_jobs.tsv"

mkdir -p "$(dirname "$RECORD_FILE")"
if [ "$DRY_RUN" != "1" ]; then
    if [ ! -f "$RECORD_FILE" ]; then
        printf 'timestamp\tjobid\tconfig\tphi_label\tphi\tseed\tvariant\n' > "$RECORD_FILE"
    fi
fi

submitted=0
skipped=0

for config in "${CONFIGS[@]}"; do
    for i in "${!PHI_LABELS[@]}"; do
        label="${PHI_LABELS[$i]}"
        phi="${PHI_VALUES[$i]}"
        for seed in "${SEEDS[@]}"; do
            variant="${STEP_TAG}-ego-eval-ood-${label}-seed${seed}"
            exp_name="${config#pi05_}_ego"
            summary="$REPO_DIR/experiments/vla_benchmark/mesa_bimanual/${exp_name}/${variant}/statistics/final_summary.json"
            if [ -f "$summary" ]; then
                echo "[skip] already done: $exp_name/$variant"
                skipped=$((skipped + 1))
                continue
            fi

            overrides="{\"egocentric\":{\"phi\":${phi},\"phi_radius\":0.2}}"

            export CONFIG="$config"
            export STEP="$STEP_TAG"
            export VARIANT_NAME="$variant"
            export CAMERA_OVERRIDES="$overrides"
            export SEED="$seed"

            if [ "$DRY_RUN" = "1" ]; then
                echo "[dry-run] $config phi=$label seed=$seed variant=$variant"
                continue
            fi

            out=$(sbatch --export=ALL "$SBATCH_SCRIPT")
            jobid=$(echo "$out" | awk '{print $NF}')
            echo "[submit] jobid=$jobid config=$config phi=$label seed=$seed variant=$variant"
            ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$ts" "$jobid" "$config" "$label" "$phi" "$seed" "$variant" \
                >> "$RECORD_FILE"
            submitted=$((submitted + 1))
        done
    done
done

echo "[summary] submitted=$submitted skipped=$skipped record=$RECORD_FILE"
