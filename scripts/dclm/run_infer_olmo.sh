begin=$1
end=$2
gpu_index=0
mkdir -p runs/infer
for ((s=begin; s<=end; s++)); do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index python scripts/dclm/run_infer_olmo.py $s configs/dclm/OLMo-300M_2x.yaml \
    > runs/infer/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done