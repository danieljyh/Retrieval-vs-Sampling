import torch
from transformers import CLIPModel, CLIPProcessor

from typing import Union
from sentence_transformers import util


class SimpleFrameSelector(torch.nn.Module):
    def __init__(
        self,
        clip_model_path="openai/clip-vit-base-patch32",
        retrieve_size=64,
    ):
        super().__init__()

        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.model = CLIPModel.from_pretrained(clip_model_path)

        self.retrieve_size = retrieve_size

    def forward(self, video, text_inputs) -> Union[torch.Tensor, torch.Tensor]:

        video_list = [video[i] for i in range(len(video))]
        inputs = self.processor(
            text=text_inputs,
            images=video_list,
            return_tensors="pt",
        )

        text_outputs = self.model.get_text_features(
            inputs["input_ids"], inputs["attention_mask"]
        ).squeeze()
        vision_outputs = self.model.get_image_features(inputs["pixel_values"])

        outputs = util.cos_sim(text_outputs, vision_outputs).squeeze()

        _, sorted_indices = torch.sort(outputs, descending=True)
        selected = sorted_indices[: self.retrieve_size]

        return video[selected]
