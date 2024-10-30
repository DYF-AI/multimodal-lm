import json
import os.path

import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:  # 遍历预定义的宽高比
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:  # 找一个最接近的宽高比值,如果相等,面积大的
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height     # 计算宽高比

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)  # 根据max_num获取可能的宽高比例
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])  # 去重

    # find the closest aspect ratio to the target, 根据原图尺寸找到一个最接近的宽高比例， 原图1000x747=1.33.。，最接近4:3
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height, 将448按照比例进行resize,得到必然是448的倍数
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]  # blocks的数量就是比例相乘,本质上类似原生vit的patch

    # resize the image, 对原图进行resize
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):  # 对图片进行crop, len(blocks)+1
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--save_result_path", default=None, type=str)
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.ckpt_dir
    test_file_path = args.val_dataset
    save_result_path = args.save_result_path

    if save_result_path is None:
        save_result_path = test_file_path.replace(".jsonl", "_pred.jsonl")

    device_map = "auto"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(
        num_beams=1,
        max_new_tokens=args.max_length,
        do_sample=False
    )
    if not os.path.exists(os.path.dirname(save_result_path)):
        os.makedirs(os.path.dirname(save_result_path))

    with open(test_file_path, "r", encoding="utf-8") as fi,\
            open(save_result_path, "w", encoding="utf-8") as fo:
        for line in tqdm(fi):
            line_data = json.loads(line)
            query, gt = line_data["query"], line_data["response"]
            image_path = line_data["image_path"][0]
            pixel_values = load_image(image_path).to(torch.bfloat16).to(device)
            question = "<image>\n" + query
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            print(response)
            row_data = {"query": query, "response": response, "images": [image_path], "gt": gt}
            fo.write(json.dumps(row_data, ensure_ascii=False) + "\n")
            fo.flush()


if __name__ == '__main__':
    args = parse_args()
    main(args)