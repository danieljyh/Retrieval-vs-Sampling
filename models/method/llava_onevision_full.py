import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger


class LlavaOneVisionFull(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config, processor, n_local):
        super().__init__(config)
        self.processor = processor
        self.n_local = n_local

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        num_frames = video.shape[0]
        for start_idx in range(0, num_frames, encode_chunk_size):
            end_idx = min(start_idx + encode_chunk_size, num_frames)
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)

    def _encode_video_chunk(self, video_chunk):
        # (1, Nv, 3, H, W)
        pixel_values_videos = self.processor.video_processor(
            video_chunk, return_tensors="pt"
        ).pixel_values_videos.to(self.device, self.dtype)

        # (1, Nv*196, D) ex. (1, 12544, 896)
        video_features = self._get_video_features(pixel_values_videos)

        assert (
            self.n_local >= video_features.shape[1]
        ), f"n_local: {self.n_local}, video_features: {video_features.shape[1]}"

        output = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(
            batch_size * frames, channels, height, width
        )
        video_features = self.vision_tower(
            pixel_values_videos, output_hidden_states=True
        )  # [(B*T, w*h, D)] * layers = [(64, 729 1152)] * 27
        selected_video_feature = video_features.hidden_states[
            self.config.vision_feature_layer
        ]  # (B*T, w*h, D) = (64, 729 1152)

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(
            video_features
        )  # (64, 729, 896) -> (64, 196, 896)
        video_features = video_features.reshape(
            batch_size, frames * video_features.shape[1], -1
        )  # (B, Nv*196, D)
        return video_features


def load_model(
    model_path="model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf",
    n_init=None,
    n_local=None,
    topk=64,
    chunk_size=1,
):
    device = "cuda"
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)

    init_prompt = (
        "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user "
    )
    init_prompt_ids = processor.tokenizer(
        init_prompt, return_tensors="pt"
    ).input_ids.to(device)

    model = LlavaOneVisionFull.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
    )

    logger.info(f"n_frame_tokens: {n_frame_tokens}")

    model.eval()

    return model, processor
