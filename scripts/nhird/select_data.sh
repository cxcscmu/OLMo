source .env

CONFIG_PATH="configs/nhird/probe-influence.yaml"
export CHECKPOINT_DIR="${LOCAL_ROOT}/out/OLMo-2-0425-1B_stage1/step1907359-vocab_expansion-midtrain/phase11_selective/step9316-unsharded/data_influence"
export CHECKPOINT_NAME=$(basename "${CHECKPOINT_DIR}")
export OUTPUT_DIR="${CHECKPOINT_DIR}/data_influence"

# Create a processed config with environment variables substituted
CONFIG_PATH="configs/nhird/probe-influence.yaml"
CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"

# Ensure cleanup on exit (success or failure)
trap "rm -f '${CONFIG_PROCESSED}'" EXIT

# 1. Extract top 20% most influential instances
python scripts/nhird/select_data.py \
    ${CONFIG_PROCESSED} \
    --output ${OUTPUT_DIR}/selection/top_influential.npy \
    --metrics-file ${CHECKPOINT_DIR}/influence.npy \
    --sample-ratio 0.2

# 2. Extract positive influential instances
python scripts/nhird/select_data.py \
    ${CONFIG_PROCESSED} \
    --output ${OUTPUT_DIR}/selection/positive_influential.npy \
    --metrics-file ${CHECKPOINT_DIR}/influence.npy \
    --sample-ratio -1

# 3. Extract specific indices (metrics-file ignored)
python scripts/nhird/select_data.py \
    ${CONFIG_PROCESSED} \
    --output ${OUTPUT_DIR}/selection/specific.npy \
    --indices-file ${OUTPUT_DIR}/selection/indices.npy

# 4. Random sampling (no metrics or indices)
python scripts/nhird/select_data.py \
    ${CONFIG_PROCESSED} \
    --output ${OUTPUT_DIR}/selection/random.npy \
    --sample-ratio 0.2
