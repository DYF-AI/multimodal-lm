from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import torch
from transformers import DonutProcessor, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForGeneration(DataCollatorMixin):
    processor: DonutProcessor
    tokenizer: DonutProcessor.tokenizer_class
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def _parse_pixel_values(self, data):
        shape = np.frombuffer(data[:12], dtype=np.int32)
        pixel_values = np.frombuffer(data[12:], dtype=np.float32).reshape(shape)
        return pixel_values

    def __call__(self, features, return_tensors="np"):
        if "pixel_values" in features[0]:
            pixel_values = np.array([self._parse_pixel_values(feature["pixel_values"]) for feature in features])
        else:
            image_list = list()
            for feature in features:
                image = feature["image"].convert("RGB")
                angle = feature["angle"]
                if angle:
                    image = image.rotate(360 - angle)
                image_list.append(image)
            random_padding = features[0]["random_padding"]
            pixel_values = self.processor(image_list, random_padding=random_padding,
                                          return_tensors=return_tensors).pixel_values
        labels = [feature["labels"] for feature in features]
        return {"pixel_values": torch.from_numpy(pixel_values), "labels": torch.tensor(labels)}
