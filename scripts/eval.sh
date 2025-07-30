export TRITON_LIBCUDA_PATH="/tools/cuda/cuda12.1/targets/x86_64-linux/lib/stubs"

# 1. Model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model="llava_ov_0.5b"

method="basemodel"
# method="rekv"
# method="sampling"

# 2. Dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
tasks=(qaego4d)
# tasks=(mlvu)

fps=0.5
retrieve_sizes=(16 32 64 128)

# ReKV, Sampling
# for retrieve_size in ${retrieve_sizes[@]}; do
#     python -m main.eval \
#         --model $model \
#         --method $method \
#         --tasks $tasks \
#         --sample_fps $fps \
#         --retrieve_size $retrieve_size \
#         --postfix "fps"$fps"_r"$retrieve_size
# done


# Basemodel
for retrieve_size in ${retrieve_sizes[@]}; do
    python -m main.eval \
        --model $model \
        --method $method \
        --tasks $tasks \
        --retrieve_size $retrieve_size \
        --postfix $retrieve_size
done