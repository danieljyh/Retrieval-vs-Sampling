export TRITON_LIBCUDA_PATH="/tools/cuda/cuda12.1/targets/x86_64-linux/lib/stubs"

# Model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model="llava_ov_7b"
method="rekv"
# method="sampling"

# Dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
tasks=(qaego4d)

fps=0.5
retrieve_size=64

python -m main.eval \
    --model $model \
    --method $method \
    --tasks $tasks \
    --sample_fps $fps \
    --retrieve_size $retrieve_size \