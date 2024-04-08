import json
from abc import ABC
import datasets
from torch.utils.data import Dataset
from transformers import DonutProcessor

from mllm.utils import json2token
from mllm.utils.image_utils import load_image, load_pil_image

"""
    原生的torch Dataset
"""


class CustomDonutFinetuneDataset(Dataset):
    def __init__(self, file_path: str, processor: DonutProcessor, split="train"):
        self.file_path = file_path
        self.processor = processor
        self.max_pathches = 256
        self.split = split
        self.data = []
        self.load_data()

    def load_data(self):
        if self.file_path.endswith(".arrow"):
            # 加载arrow文件数据, 两种方式
            self.data = datasets.Dataset.from_file(self.file_path)
            #self.data = [row for row in self.data]
        elif self.file_path.endswith(".jsonl"):
            pass
        else:
            raise ValueError("self.file_path must endswith arrow or jsonl!")

    def __getitem__(self, idx: int):
        item = self.data[idx]
        # TODO: add angle processing
        image, _ = load_pil_image(item["image"])
        encoding = self.processor(images=image,random_padding=self.split, return_tensors="pt")
        target_sequence = json2token(json.loads(item["ground_truth"]), sort_key=True) + processor.tokenizer.eos_token
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["id"] = item["id"]
        encoding["prompt"] = item["prompt"]
        encoding["ground_truth"] = item["ground_truth"]
        encoding["target_sequence"] = target_sequence
        return encoding

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    MP = "J:/model/pretrained-model/torch\donut-base"
    train_dataset_file = "J:/dataset/mllm-data/mllm-finetune-data/trainticket/train.arrow"
    processor = DonutProcessor.from_pretrained(MP)
    train_dataset = CustomDonutFinetuneDataset(train_dataset_file, processor)
    print(train_dataset[0])
