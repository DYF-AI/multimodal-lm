import logging
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image
from transformers import DonutProcessor, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
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
    """
    pytorch-lightning-donut
    :param batch:
    :param random_padding:
    :param processor:
    :return:
    """
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


class DonutCollatorForPromptV1(DataCollatorMixin):
    """
    V1版本: 同时支持prompt训练方式, 及start_token训练方式, 使用train_mode进行区分
    """
    processor: DonutProcessor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    train_mode: str = None

    def _parse_pixel_values(self, data):
        shape = np.frombuffer(data[:12], dtype=np.int32)
        pixel_values = np.frombuffer(data[12:], dtype=np.float32).reshape(shape)
        return pixel_values

    def _label_sep_process(self,
                           new_batch,
                           text_input_ids,
                           sep_token_id,
                           prompt_and_target_sequences
                           ):
        # label, 前面需要-100加持
        new_batch["labels"] = text_input_ids.input_ids.clone()
        # 找到sep_id， 然后-100, 每行都有一个sep
        for i, label in enumerate(new_batch["labels"]):
            sep_token_indexes = torch.where(label == sep_token_id)[0].data.numpy()
            if len(sep_token_indexes) == 0:
                logging.error("不应该找不到sep_token_id, prompt_and_target_sequences: %s",
                              prompt_and_target_sequences[i]
                              )
                raise ValueError()
            i_sep_token_index = int(sep_token_indexes[0])
            label[:i_sep_token_index + 1] = -100
            new_batch["labels"][i] = label
        return new_batch

    def _collator_for_prompt(self,
                             batch,
                             pixel_values,
                             sep_token="<sep/>",
                             return_tensors="pt"
                             ):
        new_batch = {"pixel_values": []}
        # 基于prompt的形式进行训练
        sep_token_id = self.tokenizer.convert_tokens_to_ids(sep_token)
        prompt_and_target_sequences = [
            feature["prompt"] + sep_token + self.tokenizer.eos_token
            for feature in batch
        ]
        # decoder_input
        text_input_ids = self.processor(
            text=prompt_and_target_sequences,
            padding=True,
            return_tensors=return_tensors,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        # 前面自带了BOS，去掉
        text_input_ids.input_ids = text_input_ids.input_ids[:, 1:]
        # decoder_input_ids,应该仅训练才用到
        new_batch["decoder_input_ids"] = shift_tokens_right(
            text_input_ids.input_ids, self.tokenizer.pad_token_id, 0
        )
        new_batch = self._label_sep_process(new_batch,
                                            text_input_ids,
                                            sep_token_id,
                                            prompt_and_target_sequences)
        prompts = [feature["prompt"] + sep_token for feature in batch]
        prompts_inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors=return_tensors,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = prompts_inputs.input_ids.new_zeros(
            (prompts_inputs.input_ids.shape[0], 1)
        )
        input_ids = torch.concat((input_ids, prompts_inputs.input_ids), dim=1)
        new_batch["input_ids"] = input_ids
        new_batch["pixel_values"] = pixel_values
        return new_batch

    def __call__(self,
                 batch,
                 return_tensors="pt",
                 sep_token="<sep/>"):
        if "pixel_values" in batch[0]:
            pixel_values = np.array(
                [self._parse_pixel_values(feature["pixel_values"]) for feature in batch]
            )
        else:
            image_list = []
            for feature in batch:
                image = Image.open(feature["image"]).convert("RGB")
                angle = feature["angle"]
                if angle:
                    image = image.rotate(360 - angle, expand=True)
                image_list.append(image)
            random_padding = batch[0]["random_padding"]
            pixel_values = self.processor(
                image_list, random_padding=random_padding, return_tensors=return_tensors
            ).pixel_values
        target_sequences = [
            feature["target_sequence"] + self.tokenizer.eos_token for feature in batch
        ]
        # 使用prompt的训练方式
        if (
                self.train_mode is not None
                and self.train_mode == "prompt"
                and batch[0].get("prompt") is not None
        ):
            new_batch = self._collator_for_prompt(batch, pixel_values, sep_token="<sep/>", return_tensors="pt")
            return new_batch

        # 以正常的start—token进行训练
        labels = self.tokenizer(
            target_sequences,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )["input_ids"]
        return {"pixel_values": pixel_values, "labels": labels}


@dataclass
class DonutCollatorForPromptV2(DataCollatorMixin):
    """
    v2版本
    """
    tokenizer: PreTrainedTokenizerBase
    sep_token: str = "<sep/>"
    max_length: int = 2560
    ignore_id: int = -100

    def __call__(self, features):
        new_features = {"pixel_values": []}
        sep_token_id = self.tokenizer.get_vocab()[self.sep_token]

        prompt_and_labels = [
            item["prompt"] + self.sep_token + item["labels"] for item in features
        ]

        text_input_ids = self.tokenizer(
            text=prompt_and_labels,
            padding=True,
            return_tensors=True,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        text_input_ids = text_input_ids.input_ids

        # decoder_input_ids, 切词器编码后删除eos_token_id
        decoder_input_ids = text_input_ids[:, :-1].clone()
        decoder_input_ids.masked_fill_(
            decoder_input_ids == self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        new_features["decoder_input_ids"] = decoder_input_ids

        # labels,寻找到sep_token后将之前的位置都置为loss_ignore_id
        labels = text_input_ids[:, 1:].clone()
        mask_lengths = torch.nonzero(labels == sep_token_id, as_tuple=False)[:, 1]
        mask_index = torch.arange(labels.shape[1]) < mask_lengths[:, None]
        labels[mask_index] = self.ignore_id
        new_features["labels"] = labels

        # for generate utils
        if features[0]["prompt"] != "":
            prompts = [item["prompt"] for item in features]
            prompts_inputs = self.tokenizer(
                text=prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )
            batch_size = prompts_inputs.input_ids.shape[0]
            # donut 就是0开始的
            input_ids = prompts_inputs.input_ids.new_zeros(batch_size, 1)
            input_ids = torch.concat((input_ids, prompts_inputs.input_ids), dim=-1)
            new_features["input_ids"] = input_ids

        new_features["pixel_features"] = torch.stack(
            [item["pixel_values"] for item in features]
        )
        return new_features
