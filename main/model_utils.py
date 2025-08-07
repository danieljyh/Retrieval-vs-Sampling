import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

from models.ReKV.llava_onevision_rekv import LlavaOneVision_ReKV
from models.method import LlavaOneVisionFull, SimpleFrameSelector
from models.ReKV.patch import patch_hf


MODEL_PATH = {
    "llava_ov_0.5b": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava_ov_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}


def load_model_and_processor(args):
    model_path = MODEL_PATH[args.model]

    if args.method == "basemodel":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        processor = LlavaOnevisionProcessor.from_pretrained(model_path)

    elif args.method == "rekv":
        # Load Processor
        processor = LlavaOnevisionProcessor.from_pretrained(model_path)

        # Load Model
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
            },
        ]
        init_prompt = processor.apply_chat_template(conversation)
        init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt")[
            "input_ids"
        ]
        n_init = init_prompt_ids.shape[1] - 1  # Delete eos token after "user"
        n_frame_tokens = 196
        n_local = 15000
        # n_local = n_frame_tokens * 64

        inf_llm_config = {
            "n_init": n_init,
            "n_local": n_local,
            "fattn": True,
            "block_size": n_frame_tokens,
            "topk": args.retrieve_size,
            "chunk_size": 1,
            "max_cached_block": 128,
            "exc_block_size": n_frame_tokens,
            "pin_memory": True,
        }

        model = LlavaOneVision_ReKV.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            processor=processor,
            n_frame_tokens=n_frame_tokens,
            init_prompt_ids=None,
            n_local=n_local,
            topk=args.retrieve_size,
            chunk_size=1,
        )
        model.language_model = patch_hf(model.language_model, **inf_llm_config)
        model.eval()

    elif args.method == "full":
        # Load Processor
        processor = LlavaOnevisionProcessor.from_pretrained(model_path)

        # Load Model
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
            },
        ]
        init_prompt = processor.apply_chat_template(conversation)
        init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt")[
            "input_ids"
        ]
        n_local = 15000

        model = LlavaOneVisionFull.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            processor=processor,
            n_local=n_local,
        )
        model.eval()

    elif args.method == "sampling":
        # Load Processor
        processor = LlavaOnevisionProcessor.from_pretrained(model_path)

        # Load Model
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
            },
        ]
        init_prompt = processor.apply_chat_template(conversation)
        init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt")[
            "input_ids"
        ]
        n_local = 15000

        lmm_model = LlavaOneVisionFull.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            processor=processor,
            n_local=n_local,
        )
        lmm_model.eval()

        sampler_model = SimpleFrameSelector(retrieve_size=args.retrieve_size)

        model = lmm_model, sampler_model

    else:
        raise NotImplementedError()

    return model, processor
