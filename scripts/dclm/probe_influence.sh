#!/bin/bash
#SBATCH --job-name=probe_influence
#SBATCH --output=logs/probe_influence_%j.out
#SBATCH --error=logs/probe_influence_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# print commands
set -x

# Probe data influence via validation-updated model (YAML-based, FSDP-compatible)
# This script uses the full train.py infrastructure with a custom config
set -euo pipefail

source .env

# Configuration
CONFIG_PATH="configs/dclm/probe-influence.yaml"
export CHECKPOINT_DIR="${LOCAL_ROOT}/out/OLMo-300M/dclm_1.4B_all_repetition/step8624-unsharded"
export CHECKPOINT_NAME=$(basename "${CHECKPOINT_DIR}")
export OUTPUT_DIR="${CHECKPOINT_DIR}/data_influence"

# Count visible GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected ${NUM_GPUS} GPUs"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi

# Create a processed config with environment variables substituted
CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"

# Ensure cleanup on exit (success or failure)
trap "rm -f '${CONFIG_PROCESSED}'" EXIT

echo "=================================================="
echo "Data Influence Computation (FSDP-compatible)"
echo "=================================================="
echo "Config: ${CONFIG_PROCESSED}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "=================================================="

# Run with torchrun
# torchrun \
#   --nproc_per_node="${NUM_GPUS}" \
#   scripts/nhird/probe_influence_v2.py \
#   "${CONFIG_PROCESSED}"
torchrun \
  --nproc_per_node="$NUM_GPUS" \
  scripts/train.py "${CONFIG_PROCESSED}"

echo "=================================================="
echo "Influence computation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="