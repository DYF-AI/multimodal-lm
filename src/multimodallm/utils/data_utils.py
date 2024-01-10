import re
import torch
import json
from typing import Any, Union, Optional
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizerBase, DonutProcessor
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy
import sys
def trans_platform(path):
    new_path = path.replace("\\", "/").replace("J:/", "/mnt/j/")
    return new_path

platform = sys.platform
def json2token(obj:Any, sort_key: bool=True):
    if isinstance(obj, list):
        return r"<sep/>".join([json2token(v, sort_key) for v in obj])
    elif isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda x:x[0]) if sort_key else obj.items()
        return "".join([fr"<s_{k}>" + json2token(v, sort_key) + fr"</s_{k}>" for k,v in items])
    obj = str(obj)
    return obj


def convert_json_key_to_id(obj, key_ids):
    if isinstance(obj, list):
        return [convert_json_key_to_id(v, key_ids) for v in obj]
    elif isinstance(obj, dict):
        new_obj = dict()
        for k, v in obj.items():
            if k not in key_ids:
                key_ids[k] = max(key_ids.values())+1
            new_obj[key_ids[k]] = convert_json_key_to_id(v, key_ids)
        return obj
    return obj


def token2json(tokens, is_inner_value=False, expand_vocab=None):
    """
    Convert a (generated) token sequence into an ordered JSON format.
    expand_vocab: 是扩展的特殊字符
    """
    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            # 换行/n 需要增加re.S
            content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE|re.S)
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True, expand_vocab=expand_vocab)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in expand_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True, expand_vocab=expand_vocab)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


def preprocess(rows, processor=None, sort_key=True, eager=False, random_padding=False, max_length=768):
    target_sequence = [json2token(json.loads(v), sort_key=sort_key)+processor.tokenizer.eos_token for v in rows["ground_truth"]]
    if platform == "linux":
        image = [trans_platform(v) for v in rows["image"]]
        rows["image"] = image


    labels = processor.tokenizer(
        target_sequence,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )["input_ids"]
    # eager：缓存图片数据， 开启缓存需要大量内存
    if not eager:
        return {
            "labels":labels,
            "target": target_sequence,
            "random_padding":[random_padding for _ in range(len(labels))]
        }
    image_list = rows["image"]
    image_angle_list = []
    for image, angle in zip(rows["image"], rows["angle"]):
        if angle:
            image = image.rotate(360-angle)
            image_angle_list.append(image)
    if len(image_angle_list) == len(image_list):
        image_list = image_angle_list
    pixel_values = processor(image_list, random_padding=random_padding, return_tensors="np").pixel_values

    return {
        "pixel_values": [np.array(v.shape, dtype=np.int32).tobytes() + v.tobytes() for v in pixel_values],
        "labels": labels,
        "target": target_sequence
    }
