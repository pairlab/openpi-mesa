#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH -J pi05_mesa_bimanual_lora_3d_gen_sanity
#SBATCH -o examples/mesa/slurm_logs/sanity_3d_gen_%j.out
#SBATCH -e examples/mesa/slurm_logs/sanity_3d_gen_%j.err
#
# Pre-training sanity checks for pi05_mesa_bimanual_lora_3d_gen: dataloader
# shape/dtype/numerics + one forward pass of Pi0Adapt3R on a real batch.
# Exits non-zero on any assertion failure. Meant to run before the 8h
# rtx6000 training job.

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/openpi-mesa"
PROJECT_DIR="/storage/project/r-agarg35-0/fchang40"
cd "$REPO_DIR"

export HF_LEROBOT_HOME="/storage/project/r-agarg35-0/shared/vla_benchmark_data/3d_data"
# HF `datasets` caches an Arrow copy of the loaded parquet under HF_HOME; keep
# that off the 20G home quota.
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venvs/openpi-mesa"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export GIT_LFS_SKIP_SMUDGE=1
# CPU-only: force JAX onto CPU even if stray GPU env is present.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch"
mkdir -p "$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "[sanity] node=$(hostname) cpu=$(nproc) platform=cpu-only"

uv run python examples/mesa/sanity_check_3d.py \
    --config-name pi05_mesa_bimanual_lora_3d_gen \
    --batch-size 2

echo "[sanity] done."
