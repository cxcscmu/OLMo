#!/bin/bash
#SBATCH --job-name=dclm_pretrain
#SBATCH --output=runs/dclm_pretrain_%j.out
#SBATCH --error=runs/dclm_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1792G
#SBATCH --time=2-00:00:00

# print commands
set -x

source .env

GCS_ROOT="gs://cmu-gpucloud-zichunyu/healthcare/olmo"
CONFIG_PATH="${1:-configs/dclm/OLMo-300M.yaml}"

mkdir -p "${LOCAL_ROOT}/data/preprocessed/"
echo "LOCAL_ROOT=${LOCAL_ROOT}"
if [[ ! -d "${LOCAL_ROOT}/data/preprocessed/dclm" ]]; then
    echo "Directory missing, downloading..."
    gcloud storage cp -r "${GCS_ROOT}/data/preprocessed/dclm" "${LOCAL_ROOT}/data/preprocessed/"
else
    echo "Directory exists, skipping download"
fi

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

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  scripts/train.py "${CONFIG_PROCESSED}"