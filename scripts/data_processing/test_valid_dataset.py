"""
        return {
  File "D:\ProgramData\Anaconda3\envs\torch\lib\site-packages\datasets\features\features.py", line 1648, in <dictcomp>
    column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
  File "D:\ProgramData\Anaconda3\envs\torch\lib\site-packages\datasets\features\features.py", line 1260, in decode_nested_example
    return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
  File "D:\ProgramData\Anaconda3\envs\torch\lib\site-packages\datasets\features\image.py", line 137, in decode_example
    image = PIL.Image.open(path)
  File "D:\ProgramData\Anaconda3\envs\torch\lib\site-packages\PIL\Image.py", line 3092, in open
    fp = builtins.open(filename, "rb")
PermissionError: [Errno 13] Permission denied: 'J:/data/mllm-data/mllm-pretrain-data/train

"""
import os
import torch
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DonutProcessor
from multimodallm.utils.data_utils import preprocess

# train_key, validation_key, test_key = "train", "validation", "test"
DP = "J:/data/mllm-data/mllm-pretrain-data"

train_dataset = "J:/data/mllm-data/mllm-pretrain-data/train.arrow"
validation_dataset = "J:/data/mllm-data/mllm-pretrain-data/validation.arrow"
test_dataset = "J:/data/mllm-data/mllm-pretrain-data/test.arrow"
max_length = 2560
image_size = [1024, 1024]
MP = "J:/model/mllm-model/donut-pretrain/20231124/pl-checkpoint-14500-ned-0.8225900701319295"
processor = DonutProcessor.from_pretrained(MP)
processor.image_processor.size = image_size[::-1]  # should be (width, height)
processor.image_processor.do_align_long_axis = False

en_ds = {}
for dataset_path in [train_dataset, validation_dataset, test_dataset]:
    dataset = datasets.Dataset.from_file(dataset_path)  # .select(range(10))
    key = os.path.basename(dataset_path).split(".")[0]
    if "_cache" in key: key = key.replace("_cache", "")
    # "eager":True, 缓存图片数据到内存，但是非常耗内存
    en_ds[key] = dataset.map(preprocess,
                             fn_kwargs={"processor": processor,
                                        "sort_key": False,
                                        "eager": False,
                                        "random_padding": key == "train",
                                        "max_length": max_length},
                             batched=True,
                             batch_size=64,
                             keep_in_memory=True)
print(en_ds)
print(en_ds["train"][0])


def custom_collate(batch, random_padding):
    ids, images, angles, ground_truths, labels, targets, random_paddings = [], [], [], [], [], [], []
    for sample in batch:
        ids.append(sample["id"])
        images.append(sample["image"].convert("RGB"))
        angles.append(sample["angle"])
        ground_truths.append(sample["ground_truth"])
        labels.append(sample["labels"])
        targets.append(sample["target"])
        random_paddings.append(sample["random_padding"])
    pixel_values = processor(images, random_padding=random_padding,
                             return_tensors="pt").pixel_values  # .squeeze()
    return ids, pixel_values, angles, ground_truths, torch.tensor(labels), targets, random_paddings


train_dataloader = DataLoader(en_ds["train"],
                              batch_size=1,
                              shuffle=True,
                              collate_fn=lambda x: custom_collate(x, True),
                              num_workers=0)
val_dataloader = DataLoader(en_ds["validation"],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: custom_collate(x, False),
                            num_workers=0)
# for batch in tqdm(train_dataloader):
#     batch = next(iter(train_dataloader))

error_nums = 0
error_list = list()

for i, data in tqdm(enumerate(train_dataloader)):
    try:
        print(i, error_nums)
    except:
        error_nums += 1
        print(i, data)
        error_list.append(data)
print(error_list)