import torch


def sampling_generate(
    model,
    processor,
    data,
    video,
    generation_config,
):
    return 0


def rekv_generate(
    model,
    processor, 
    data,
    video,
    generation_config,
):
    """
    Inputs:
        data: Dict
            {
                "video_id": "BV1s94y1G7RD",
                "video_path": "data/cgbench/videos/BV1s94y1G7RD.mp4",
                "duration": 1980,
                "conversations": [
                    {
                        "question": "Where did I put the dog fur?",
                        "choices": [
                            "on the sofa",
                            "on the floor",
                            "on the table",
                            "in the trash
                        ],
                        "answer": "on the sofa",
                        "answer_letter": "A", # if the task is multiple choice
                        "prompt": "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user <video>\nQuestion: Where did I put the dog fur?\nOptions:\n(A) on the sofa\n(B) on the floor\n(C) on the table\n(D) in the trash\nOnly give the best option.<|im_end|><|im_start|>assistant\nBest option: (" 
                        "temporal_windows": [
                            [
                                5,
                                7
                            ]
                        ]
                    },
                    ...,
                ]
            }
        
        video: np.array (Frame, H, W, Channel)

        generation_config: Dict
            {
                "max_new_tokens": Int,
                "eos_token_ids": List[token_id]     
            }
    

    Outputs:
        List[Dict]
        {
            "pred": pred,
            "answer_letter": sample["answer_letter"], # if the task is multiple choice
            "answer": sample["answer"],
            "question": sample["question"],
            "choices": sample["choices"], # if the task is multiple choice
            "video_id": data["video_id"],
        }
    """


    # 1. Prefill Phase
    model.clear_cache()
    init_prompt = data["conversations"][0]["prompt"].split("<video>")[0]
    inputs = processor.tokenizer(init_prompt, return_tensors="pt").to(model.device)
    outputs = model.language_model(input_ids=inputs["input_ids"], use_cache=True, return_dict=True)
    model.kv_cache = outputs.past_key_values
    model.encode_video(video)

    results = []
    for sample in data['conversations']:
        question = sample['question']
        prompt = sample["prompt"].split("<video>")[1]
        
        
        # 2. Retrieval Phage
        # NOTE: Only input the question to perform retrieval.
        inputs = processor.tokenizer(question, return_tensors="pt").to(model.device)

        for layer_kv in model.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        outputs = model.language_model(
            input_ids=inputs["input_ids"], 
            use_cache=True, 
            past_key_values=model.kv_cache
        )
        past_key_values = outputs.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in model.kv_cache:  # reset to default
            layer_kv.reset_retrieval()


        # 3. Decoding Phase
        output_ids = []
        eos_token_ids = generation_config["eos_token_ids"]
        max_new_tokens = generation_config["max_new_tokens"] 
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                inputs = processor.tokenizer(prompt, return_tensors="pt").to(model.device)
                inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
                outputs = model.language_model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=past_key_values
                )

            else:  # decoding
                outputs = model.language_model(
                    input_ids=torch.as_tensor([[token]], device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in eos_token_ids:
                break

        
        # 4. Results
        pred = processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        results.append({
            "pred": pred,
            "answer_letter": sample["answer_letter"],
            "answer": sample["answer"],
            "question": sample["question"],
            "choices": sample["choices"], # if the task is multiple choice
            "video_id": data["video_id"],
        })
        
    return results            