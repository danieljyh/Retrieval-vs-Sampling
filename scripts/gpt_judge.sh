# 1. Model
# model="llava_ov_0.5b"
model="llava_ov_7b"

method="basemodel"
# method="rekv"
# method="sampling"

# 2. Dataset
task="videoeval_pro"

# 3. Variants
fps=0.5
retrieve_sizes=(1 4 8 16 32 64)


for retrieve_size in ${retrieve_sizes[@]}; do
    input_path="./results/"$model/$method/${task}_${retrieve_size}.jsonl
    # input_path="./results/"$model/$method/${task}_${fps}fps_r${retrieve_size}.jsonl

    output_path="./results/"$model/$method/${task}_${retrieve_size}_judged.jsonl
    # output_path="./results/"$model/$method/${task}_${fps}fps_r${retrieve_size}_judged.jsonl

    python main/gpt4o_judge.py \
        --input_path $input_path \
        --output_path $output_path \
        --model_name "gpt-4o-2024-08-06" \
        --num_threads 1
done