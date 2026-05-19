#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --cpus-per-task=2
#SBATCH -J pi05_mesa_smoke_eval_bddls
#SBATCH -o examples/mesa/slurm_logs/smoke_eval_bddls_%j.out
#SBATCH -e examples/mesa/slurm_logs/smoke_eval_bddls_%j.err
#
# CPU-only smoke for the vla-benchmark BDDL/init-state pairing. Mirrors what
# ``eval_server_parallel.py`` does pre-policy: build the env, call
# ``env.reset_to(init_state)``. If this passes, the L40S eval job will at
# least make it past episode init for every (task, split, instance_idx) we
# checked — the failure mode behind the 2026-04-26 train-split incident.
#
# Knobs:
#   EVAL_SET_NAME   default: mesa_bimanual.
#   SPLITS          default: "eval overfit".
#   TASKS           default: "apple_tray_on lime_bowl_on".
#   NUM_INSTANCES   default: 5 (smokes idx 0..N-1 per split).
#   CAMERA_NAMES    default: "egocentric leftshoulder rightshoulder midshoulder".

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

EVAL_SET_NAME="${EVAL_SET_NAME:-mesa_bimanual}"
SPLITS="${SPLITS:-eval overfit}"
TASKS="${TASKS:-apple_tray_on lime_bowl_on}"
NUM_INSTANCES="${NUM_INSTANCES:-5}"
CAMERA_NAMES="${CAMERA_NAMES:-egocentric leftshoulder rightshoulder midshoulder}"

VLA_BENCHMARK_PYTHON="${VLA_BENCHMARK_PYTHON:-$PROJECT_DIR/venvs/vla-benchmark/bin/python}"

# EGL is the right headless backend on cpu-small: ``mujoco.gl_context``
# eagerly imports the backend module at ``import robosuite`` time, and the
# venv's OSMesa wheel can't bind libGL on these nodes (PyOpenGL platform
# `None`). EGL only needs the libEGL stub present; we never call render() in
# the smoke path, so no real GPU context is created.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$REPO_DIR/examples/mesa/slurm_logs"

echo "[smoke-eval-bddls] node=$(hostname) eval_set=$EVAL_SET_NAME splits=$SPLITS"
echo "[smoke-eval-bddls] tasks=$TASKS num_instances=$NUM_INSTANCES cameras=$CAMERA_NAMES"

"$VLA_BENCHMARK_PYTHON" examples/mesa/smoke_test_eval_bddls.py \
    --eval-set "$EVAL_SET_NAME" \
    --splits $SPLITS \
    --tasks $TASKS \
    --num-instances "$NUM_INSTANCES" \
    --camera-names $CAMERA_NAMES
