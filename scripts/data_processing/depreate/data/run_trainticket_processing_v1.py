# -*-coding:utf-8 -*-
"""
    火车票数据处理
    1. processing_meta_data: label -> meta_data
    2. processing_arrow_data: meta_data -> arrow_data
"""
import json
import jsonlines
import os
import datasets
from PIL import Image
from datasets import Dataset, DatasetDict
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from mllm.utils.image_utils import load_image

"""
    #TODO
    这个代码由bug, writer.write(example, file_name) 无法写入数据 
    带排查， 以替换为 run_trainticket_processing_v2.py
"""

def processing_arrow_data(metadata_file: str, save_arrow_path: str):
    dataset_features = datasets.Features(
        {
            "id": datasets.Value("string"),
            #"image": datasets.features.Image(),  # 路径
            "image": datasets.Value("string"),  # 路径
            "angle": datasets.Value("float"),
            "ground_truth": datasets.Value("string"),
        }
    )
    image_root = os.path.dirname(metadata_file)
    save_arrow_path = save_arrow_path.replace("\\", "/")
    writer = ArrowWriter(features=dataset_features, path=save_arrow_path,  hash_salt="zh")
    with jsonlines.open(metadata_file) as f1:
        for line_data in tqdm(f1):
            try:
                #line_data = json.loads(line)
                file_name = line_data["file_name"]
                image_path = os.path.join(image_root, file_name).replace("\\", "/")
                record = {
                    "id": file_name,
                    #"image":  Image.open(image_path).convert("RGB"),
                    "image":  image_path,
                    "angle": line_data.get("angle", 0),
                    "ground_truth": json.dumps(json.loads(line_data["text"]), ensure_ascii=False)
                }
                example = dataset_features.encode_example(record)
                print(example)
                writer.write(example, file_name)
            except Exception as e:
                print(e)
        writer.close()


if __name__ == "__main__":
    train_metadata_file = "J:/dataset/mllm-data/mllm-finetune-data/trainticket/train/metadata.jsonl"
    test_metadata_file = "J:/dataset/mllm-data/mllm-finetune-data/trainticket/test/metadata.jsonl"
    save_arrow_root = "J:/dataset/mllm-data/mllm-finetune-data/trainticket"

    processing_arrow_data(train_metadata_file, os.path.join(save_arrow_root, "train.arrow"))
    processing_arrow_data(test_metadata_file, os.path.join(save_arrow_root, "test.arrow"))

    train_arrow_dataset = Dataset.from_file(os.path.join(save_arrow_root, "train.arrow"))
    test_arrow_dataset = Dataset.from_file(os.path.join(save_arrow_root, "test.arrow"))

    dataset = DatasetDict({"train":train_arrow_dataset, "test":test_arrow_dataset})