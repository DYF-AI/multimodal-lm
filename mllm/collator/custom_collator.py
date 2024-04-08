from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image
from transformers import DonutProcessor, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

"""
    1.huggingface的collator如果如下传入更多的参数,需要构造成类的格式
       self.trainer = GenSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=self._data_collator,  # self._data_collator是一个实例的类
            compute_metrics=self.computer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    2.pytorch-lightning的collator一般是在dataloadr,支持传入更多的参数, 如下所示
      - train_dataloader = DataLoader(train_dataset_select,
                                      batch_size=all_train_config[task_name]["per_device_train_batch_size"],
                                      shuffle=True,
                                      collate_fn=lambda x: custom_collate(x, True, processor),
                                      num_workers=0)
"""

def custom_collate(batch, random_padding, processor):
    ids, images, angles, ground_truths, labels, targets, random_paddings = [], [], [], [], [], [], []
    for sample in batch:
        ids.append(sample["id"])
        images.append(Image.open(sample["image"]).convert("RGB"))
        angles.append(sample["angle"])
        ground_truths.append(sample["ground_truth"])
        labels.append(sample["labels"])
        targets.append(sample["target"])
        random_paddings.append(sample["random_padding"])
    pixel_values = processor(images, random_padding=random_padding,
                             return_tensors="pt").pixel_values  # .squeeze()
    return ids, pixel_values, angles, ground_truths, torch.tensor(labels), targets, random_paddings

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
                image = Image.open(feature["image"]).convert("RGB")
                angle = feature["angle"]
                if angle:
                    image = image.rotate(360 - angle)
                image_list.append(image)
            random_padding = features[0]["random_padding"]
            pixel_values = self.processor(image_list, random_padding=random_padding,
                                          return_tensors=return_tensors).pixel_values
        labels = [feature["labels"] for feature in features]
        return {"pixel_values": torch.from_numpy(pixel_values), "labels": torch.tensor(labels)}


@dataclass
class DataCollatorForPromptGeneration(DataCollatorMixin):
    pass
