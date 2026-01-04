# 1. Extract top 20% most influential instances
python scripts/nhird/extract_by_indices.py \
    configs/nhird/probe-influence.yaml \
    --output /path/to/top_influential.npy \
    --metrics-file /path/to/influence_scores.npy \
    --sample-ratio 0.2

# 2. Extract specific indices (metrics-file ignored)
python scripts/nhird/extract_by_indices.py \
    configs/nhird/probe-influence.yaml \
    --output /path/to/specific.npy \
    --indices-file /path/to/indices.npy

# 3. Random sampling (no metrics or indices)
python scripts/nhird/extract_by_indices.py \
    configs/nhird/probe-influence.yaml \
    --output /path/to/random.npy \
    --sample-ratio 0.2
