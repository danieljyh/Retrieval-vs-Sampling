import os
import json
import torch
import pathlib
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import HfArgumentParser


from .arguments import Args
from .dataset_utils import (
    DATASET2PROMPT,
    DATASET2MAXNEWTOKENS,
    DATASET2CATEGORY,
    scorer,
)
from .model_utils import load_model_and_processor
from .model_generate import rekv_generate, sampling_generate, full_generate


CHOICE_LETTER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def main():
    # Parse
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

    # 1. Load Model & Processor
    print("Loading Model & Processor...")
    model, processor = load_model_and_processor(args)

    # Generation config
    eos_token_id = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if args.newline_as_eos:
        eos_token_id.append(
            processor.tokenizer.encode("\n", add_special_tokens=False)[-1]
        )
    generation_config = {"eos_token_ids": eos_token_id}

    # 2. Model Generate
    result_dir = os.path.join(args.result_dir, args.model, args.method)
    # result_dir = os.path.join(result_dir, args.prefix_dir) if args.prefix_dir else result_dir
    for i, task in enumerate(args.tasks):
        if args.load_results:
            results = []
            results_path = os.path.join(
                args.result_dir, args.model, args.method, f"{task}.jsonl"
            )
            with open(results_path, "r") as f:
                for line in f:
                    results.append(json.loads(line))
        else:
            print(f"Evaluating {task} ({i + 1} / {len(args.tasks)})...")

            # Load dataset
            dataset_path = os.path.join(args.dataset_dir, task, "test.json")
            dataset = json.load(open(dataset_path))

            results = []
            for data in tqdm(dataset):
                # Format data
                for sample in data["conversations"]:
                    if DATASET2CATEGORY[task] == "multiple_choice":
                        choices = sample["choices"]
                        sample["answer_letter"] = CHOICE_LETTER[
                            choices.index(sample["answer"])
                        ]

                        formatted_choices = "\n".join(
                            [
                                "(" + CHOICE_LETTER[i] + ") " + choice
                                for i, choice in enumerate(choices)
                            ]
                        )
                        formatted_question = f"Question: {sample['question']}\nOptions:\n{formatted_choices}\nOnly give the best option."

                    elif DATASET2CATEGORY[task] == "open_ended":
                        formatted_question = sample["question"]

                    conversation = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video"},
                                {"type": "text", "text": formatted_question},
                            ],
                        },
                    ]
                    sample["prompt"] = processor.apply_chat_template(
                        conversation, add_generation_prompt=True
                    )
                    if DATASET2CATEGORY[task] == "multiple_choice":
                        sample["prompt"] += "Best option: ("

                # Load video
                video_path = data["video_path"]
                vr = VideoReader(video_path, ctx=cpu(0))
                fps = round(vr.get_avg_fps())
                frame_idx = [i for i in range(0, len(vr), int(fps / args.sample_fps))]
                video = vr.get_batch(frame_idx).asnumpy()

                # Generate
                max_new_tokens = DATASET2MAXNEWTOKENS[task]
                generation_config["max_new_tokens"] = max_new_tokens

                if args.method == "rekv":
                    results.extend(
                        rekv_generate(model, processor, data, video, generation_config)
                    )
                elif args.method == "sampling":
                    results.extend(
                        sampling_generate(
                            model, processor, data, video, generation_config
                        )
                    )
                elif args.method == "full":
                    results.extend(
                        full_generate(model, processor, data, video, generation_config)
                    )
                else:
                    raise NotImplementedError()

        # 3. Evaluate & Save
        score = scorer(task, results)
        print(f"{task}: {score}")

        file_name = f"{task}_{args.postfix}.jsonl" if args.postfix else f"{task}.jsonl"
        result_path = os.path.join(result_dir, file_name)
        with open(makedirs(result_path), "w", encoding="utf-8") as f:
            f.write(json.dumps(score, ensure_ascii=False) + "\n")
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
