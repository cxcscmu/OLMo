#!/bin/bash
#SBATCH --job-name=nhird_selective
#SBATCH --partition=cx-hyper-p
#SBATCH --output=runs/nhird_selective_%j.out
#SBATCH --error=runs/nhird_selective_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=1536G
#SBATCH --time=2-00:00:00

# Multi-Phase Training Script
# This script sequentially runs multiple training phases with different data mixtures.
# Each phase loads the checkpoint from the previous phase.

# print commands
set -x

source .env

GCS_ROOT="gs://cmu-gpucloud-zichunyu/healthcare/olmo"

# Ensure local directories exist
mkdir -p "${LOCAL_ROOT}"
if [[ ! -d "${LOCAL_ROOT}/data" ]]; then
  gcloud storage cp -r "${GCS_ROOT}/data" ${LOCAL_ROOT}
fi
if [[ ! -d "${LOCAL_ROOT}/out" ]]; then
  gcloud storage cp -r "${GCS_ROOT}/out" ${LOCAL_ROOT}
fi

# Count visible GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected ${NUM_GPUS} GPUs"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi

# Function to run a single phase
run_phase() {
  local phase_num=$1
  local config_path="configs/nhird/8phases_selective/phase${phase_num}.yaml"

  echo "=================================================="
  echo "Starting Phase ${phase_num}"
  echo "Config: ${config_path}"
  echo "=================================================="

  # Create a processed config with environment variables substituted
  CONFIG_PROCESSED="${config_path}_$$.yaml"
  envsubst < "${config_path}" > "${CONFIG_PROCESSED}"

  # Ensure cleanup on exit (success or failure)
  trap "rm -f '${CONFIG_PROCESSED}'" EXIT

  # Run training
  torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=$((12345 + phase_num)) \
    scripts/train.py "${CONFIG_PROCESSED}"

  local exit_code=$?
  rm -f "${CONFIG_PROCESSED}"

  if [ $exit_code -ne 0 ]; then
    echo "ERROR: Phase ${phase_num} failed with exit code ${exit_code}"
    exit $exit_code
  fi

  echo "=================================================="
  echo "Phase ${phase_num} completed successfully"
  echo "=================================================="
  echo ""
}

# Run all phases sequentially
for phase in 12; do
  run_phase $phase
done