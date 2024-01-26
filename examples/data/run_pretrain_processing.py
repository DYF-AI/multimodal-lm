
import json
import os

import numpy as np
from tqdm import tqdm
import pandas as pd
import datasets
from datasets.arrow_writer import ArrowWriter

from multimodallm.ocr_system.ocr_info import PicInfo
from multimodallm.utils.data_utils import preprocess


"""
    预训练数据处理
    1. processing_meta_data: label -> meta_data
    2. processing_arrow_data: meta_data -> arrow_data
"""

from multimodallm.utils.file_utils import getAllFiles


def get_ocr_rows_text(ocr_rows):
    rows_text = list()
    for row in ocr_rows:
        row_text = [box.bbox_text for box in row]
        rows_text.append(" ".join(row_text))
    # 在donut-tokenizer中"\n"会被处理成空格, 需额外增加特定的换行符号"</n>"
    return "</n>".join(rows_text)

def ocr_res_pop(ocr_rows, points_num=2):
    new_ocr_res = []
    for row in ocr_rows:
        for box in row:
            box.pop("difficult")
            box.pop("score")
            if points_num == 2:
                # 仅使用两个点
                box["points"] = [box["points"][0], box["points"][2]]
            new_ocr_res.append(box)
    return new_ocr_res


def get_label_files(data_root:str):
    usages = ["train", "validation", "test"]
    label_files = []
    for usage in tqdm(usages):
        usage_root = os.path.join(data_root, usage)
        for type in os.listdir(usage_root):
            type_root = os.path.join(usage_root, type)
            label_files.append(os.path.join(type_root, "Label.txt"))
    return label_files

def adapte_ocr_data(ocr_res, file_id):
    ocr_res_data = json.loads(ocr_res)
    adapter_data = {
        "file_id": file_id,
        "bboxes": ocr_res_data,
    }
    return adapter_data

def processing_meta_data(data_root: str, save_meta_data: str):
    meta_data = {
        "图片绝对路径": list(),
        "图片相对路径": list(),
        "ocr结果": list(),
        "ocr成行": list(),
        "类型": list(),
        "用途": list()
    }
    # txt_files = getAllFiles(data_root, [".txt"])
    # label_files = [file for file in txt_files if "Label.txt" in file]
    label_files = get_label_files(data_root)
    print(f"{len(label_files)}")
    for label_file in tqdm(label_files):
        print(f"label_file:{label_file}")
        with open(label_file, "r", encoding="utf-8") as f1:
            image_root = label_file.rsplit("\\", 2)[0]
            label_file_split = label_file.split("\\")
            usage, type = label_file_split[4], label_file_split[5]
            for sample in tqdm(f1):
                file_id, ocr_res = sample.strip().split("\t")
                absolute_image_path = os.path.join(image_root, file_id).replace("\\", "/")
                relative_image_path = absolute_image_path.split("/", 3)[3]
                if not os.path.isfile(absolute_image_path):
                    continue
                meta_data["图片绝对路径"].append(absolute_image_path)
                meta_data["图片相对路径"].append(relative_image_path)
                picInfo = PicInfo.from_ocr_res(adapte_ocr_data(ocr_res, file_id))
                meta_data["ocr结果"].append(ocr_res)
                ocr_rows_text = get_ocr_rows_text(picInfo.rows)
                meta_data["ocr成行"].append(ocr_rows_text)
                meta_data["类型"].append(type)
                meta_data["用途"].append(usage)
    df = pd.DataFrame(meta_data)
    df.to_csv(save_meta_data, index=False)


def processing_arrow_data(meta_data_file:str, save_arrow_row:str):
    dataset_features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "image": datasets.Value("string"), #datasets.features.Image(),  # 该为string, 支持win和linux平台
            "angle": datasets.Value("float"),
            #"ocr_box": datasets.Value("string"),
            "ground_truth": datasets.Value("string"),   # ocr
        }
    )
    train_arrow_path = os.path.join(save_arrow_row, "train.arrow")
    validation_arrow_path = os.path.join(save_arrow_row, "validation.arrow")
    test_arrow_path = os.path.join(save_arrow_row, "test.arrow")
    train_writer = ArrowWriter(features=dataset_features, path=train_arrow_path)
    validation_writer = ArrowWriter(features=dataset_features, path=validation_arrow_path)
    test_writer = ArrowWriter(features=dataset_features, path=test_arrow_path)
    df = pd.read_csv(meta_data_file)
    print(df)
    for index, row in tqdm(df.iterrows()):
        print(index)
        file_name, usage = row["图片相对路径"], row["用途"]
        usage =  row["用途"]
        if not os.path.isfile(row["图片绝对路径"]):
            continue
        record = {
            "id": file_name,
            "image": row["图片绝对路径"],
            "angle": row.get("angle", 0),
            #"ground_truth": json.dumps({"gt_parse":{"text_sequence": row["ocr成行"]}}, ensure_ascii=False)
            "ground_truth": json.dumps({"text_sequence": row["ocr成行"]}, ensure_ascii=False)
        }
        example = dataset_features.encode_example(record)
        if usage == "train":
            train_writer.write(example, file_name)
        elif usage == "validation":
            validation_writer.write(example, file_name)
        elif usage == "test":
            test_writer.write(example, file_name)
        else:
            raise ValueError(f"usage must in ['train', 'validation', 'test']")
    train_writer.close()
    validation_writer.close()
    test_writer.close()



if __name__ == "__main__":
    data_root = r"J:\dataset\mllm-data\mllm-pretrain-data"
    save_meta_data = "J:\dataset\mllm-data\mllm-pretrain-data\mllm-data-20231116.csv"
    gen_meta_data, gen_arrow_data = True, True
    if gen_meta_data:
        processing_meta_data(data_root, save_meta_data)

    if gen_arrow_data:
        processing_arrow_data(save_meta_data, data_root)

    train_key, validation_key, test_key = "train", "validation", "test"
    MP = r"J:\model\pretrained-model\torch\donut-base"
    image_size = [1024, 1024]
    from transformers import DonutProcessor
    processor = DonutProcessor.from_pretrained(MP)
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis=False

    en_ds = {}
    for key in [train_key, validation_key, test_key]:
        train_dataset = datasets.Dataset.from_file(os.path.join(data_root, f"{key}.arrow"))
        train_dataset.map(preprocess, fn_kwargs={"processor":processor,"sort_key":False, "random_padding":key=="train",
                                                 "max_length":768}, batched=True, batch_size=16, keep_in_memory=True)
        en_ds[key]=train_dataset
    print(en_ds)
