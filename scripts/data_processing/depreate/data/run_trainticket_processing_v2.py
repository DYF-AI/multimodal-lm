# -*- coding:utf-8 -*-
import os
import json
import datasets
from PIL import Image
import numpy as np
from datasets import load_dataset
from datasets.arrow_writer import ArrowWriter
from datasets import Dataset, DatasetDict

from mllm.utils.image_utils import load_image

dataset_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        #"image": datasets.features.Image(),
        "image": datasets.Value("string"),
        "angle": datasets.Value("float"),
        "ground_truth": datasets.Value("string"),
    }
)

def generate_example(metadata_path:str, image_path:str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            print(sample)
            file_name = sample["file_name"]
            image_file = os.path.join(image_path, file_name)
            #original_image, size = load_image(image_file, return_chw=False, return_bgr=False)
            record = {
                "id": file_name,
                #"image": image_file,
                "image": image_file,
                "angle": sample.get("angle", 0),
                "ground_truth": json.dumps(json.loads(sample["text"]), ensure_ascii=False)
            }
            yield file_name, record


def build_data(metadata_path:str, image_path:str, output_path:str):
    writer = ArrowWriter(features=dataset_features,
                         path=output_path,
                         # hash_salt="zh"
                         )
    it = generate_example(metadata_path, image_path)
    try:
        for key, record in it:
            example = dataset_features.encode_example(record)
            writer.write(example, key)
    finally:
        num_examples, num_bytes = writer.finalize()
        writer.close()



if __name__ == "__main__":
    #train_image_path = "/mnt/j/datasets/document-intelligence/EATEN数据集/dataset_trainticket/train/hcp_aug_2"
    train_image_path = "J:/datasets/mllm-data/mllm-finetune-data/trainticket/train"
    save_arrow_root = "J:/datasets/mllm-data/mllm-finetune-data/trainticket"
    train_metadata_path = os.path.join(train_image_path, "metadata.jsonl")
    train_output_path = os.path.join(save_arrow_root, "train_cache.arrow")

    #test_image_path = "/mnt/j/datasets/document-intelligence/EATEN数据集/dataset_trainticket/test"
    test_image_path = "J:/datasets/mllm-data/mllm-finetune-data/trainticket/test"
    test_metadata_path = os.path.join(test_image_path, "metadata.jsonl")
    test_output_path = os.path.join(save_arrow_root, "test_cache.arrow")

    build_data(train_metadata_path, train_image_path, train_output_path)
    build_data(test_metadata_path, test_image_path, test_output_path)

    train_arrow_dataset = Dataset.from_file(train_output_path)
    test_arrow_dataset = Dataset.from_file(test_output_path)

    dataset = DatasetDict({"train":train_arrow_dataset, "test":test_arrow_dataset})

    print(dataset)