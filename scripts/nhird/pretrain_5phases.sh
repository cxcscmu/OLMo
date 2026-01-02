#!/bin/bash
#SBATCH --job-name=nhird_5phases
#SBATCH --output=logs/nhird_5phases_%j.out
#SBATCH --error=logs/nhird_5phases_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=208
#SBATCH --mem=1792G
#SBATCH --time=1-00:00:00

# 5-Phase Training Script
# This script sequentially runs 5 training phases with different data mixtures.
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
  local config_path="configs/nhird/5phases/phase${phase_num}.yaml"

  echo "=================================================="
  echo "Starting Phase ${phase_num}/5"
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
    scripts/train.py "${CONFIG_PROCESSED}"

  local exit_code=$?
  rm -f "${CONFIG_PROCESSED}"

  if [ $exit_code -ne 0 ]; then
    echo "ERROR: Phase ${phase_num} failed with exit code ${exit_code}"
    exit $exit_code
  fi

  echo "=================================================="
  echo "Phase ${phase_num}/5 completed successfully"
  echo "=================================================="
  echo ""
}

# Run all 5 phases sequentially
for phase in {1..5}; do
  run_phase $phase
done

echo "=================================================="
echo "All 5 phases completed successfully!"
echo "=================================================="