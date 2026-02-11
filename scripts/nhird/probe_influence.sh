#!/bin/bash
set -euo pipefail

source .env

# Configuration
CONFIG_PATH="configs/nhird/probe-influence.yaml"
export CHECKPOINT_DIR="${LOCAL_ROOT}/out/OLMo-2-0425-1B_stage1/step1907359-vocab_expansion-midtrain/phase12_selective/step9624-unsharded"
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
torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port=12345 \
  scripts/nhird/probe_influence_v2.py \
  "${CONFIG_PROCESSED}"

echo "=================================================="
echo "Influence computation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="