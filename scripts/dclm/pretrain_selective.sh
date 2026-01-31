#!/bin/bash
#SBATCH --job-name=nhird_5phases
#SBATCH --partition=cx-hyper-p
#SBATCH --output=logs/nhird_5phases_%j.out
#SBATCH --error=logs/nhird_5phases_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=1536G
#SBATCH --time=2-00:00:00

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
  local config_path="configs/dclm/OLMo-300M_selective_2x.yaml"

  echo "=================================================="
  echo "Starting Phase ${phase_num} training"
  echo "Config: ${config_path}"
  echo "=================================================="

  # Create a processed config with environment variables substituted
  # Determine checkpoint directory based on phase number
  if [ "$phase_num" -eq 20 ]; then
    export CHECKPOINT_DIR="${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_all_repetition/step43680-unsharded"
  else
    local prev_phase=$((phase_num - 1))
    export CHECKPOINT_DIR="${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_selective_repetition_gumbel/phase_${prev_phase}/latest-unsharded"
  fi
  export OUTPUT_DIR="${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_selective_repetition_gumbel/phase_${phase_num}"
  CHECKPOINT_NAME=$(basename "${CHECKPOINT_DIR}")
  export DATA_PATH="${LOCAL_ROOT}/data/preprocessed/dclm/${CHECKPOINT_NAME}_selection/train_ids_olmo_gumbel.npy"
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
  echo "Phase ${phase_num} training completed successfully"
  echo "=================================================="

  # Run probe influence to generate data for next iteration
  echo ""
  echo "=================================================="
  echo "Running probe_influence for Phase ${phase_num}"
  echo "=================================================="

  bash scripts/dclm/probe_influence.sh "out/OLMo-300M/dclm_2.3B_selective_repetition_gumbel/phase_${phase_num}/latest-unsharded"

  local probe_exit_code=$?
  if [ $probe_exit_code -ne 0 ]; then
    echo "ERROR: probe_influence for Phase ${phase_num} failed with exit code ${probe_exit_code}"
    exit $probe_exit_code
  fi

  echo "=================================================="
  echo "probe_influence for Phase ${phase_num} completed"
  echo "=================================================="
  echo ""
}

# Starting phase number (can be modified to resume from any phase)
START_PHASE=20

for phase in $(seq $START_PHASE 25); do
  run_phase $phase
done