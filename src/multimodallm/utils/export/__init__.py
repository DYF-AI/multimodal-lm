import os
import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig


def load_checkpoint(model_path,
                     image_size=None,
                     max_length=768,
                     task_prompt="s_ocr_pretrain"):
    with torch.no_grad():
        if image_size is None:
            image_size = [1280, 960]
        config = VisionEncoderDecoderConfig.from_pretrained(model_path)
        config.encoder.image_size = image_size # (height, width)
        config.decoder.max_length = max_length


