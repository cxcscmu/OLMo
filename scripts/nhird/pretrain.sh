#!/bin/bash
#SBATCH --job-name=nhird_midtrain
#SBATCH --output=logs/nhird_midtrain_%j.out
#SBATCH --error=logs/nhird_midtrain_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1792G
#SBATCH --time=2-00:00:00

# print commands
set -x

source .env

GCS_ROOT="gs://cmu-gpucloud-zichunyu/healthcare/olmo"
CONFIG_PATH="${1:-configs/nhird/OLMo2-1B-stage2-1ep.yaml}"

mkdir -p "${LOCAL_ROOT}"
if [[ ! -d "${LOCAL_ROOT}/data" ]]; then
  gcloud storage cp -r "${GCS_ROOT}/data" ${LOCAL_ROOT}
fi
if [[ ! -d "${LOCAL_ROOT}/pretrained_ckpt" ]]; then
  gcloud storage cp -r "${GCS_ROOT}/pretrained_ckpt" ${LOCAL_ROOT}
fi

torchrun --nproc_per_node=8 scripts/train.py "${CONFIG_PATH}"
