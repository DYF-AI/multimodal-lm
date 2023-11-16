import os
from tqdm import tqdm
import pandas as pd
import datasets
from datasets.arrow_writer import ArrowWriter

"""
    将数据：J:\data\mllm-data下所有的标注Label数据，保存为mllm-data-20231114.csv,
    需配置各类数据集合的权重，这样才能更好查看各类任务的训练情况
    比例：
        image-book: 0.95, 0.025, 0.025, 数据集太大了, 不宜分配过多的验证集、测试集
        other：0.90， 0.05， 0.05
    step1：统计各类数据的分布
    step2：根绝分布计算:train_num, val_num, test_num
    step3: 更新 用途
"""

from multimodallm.utils.file_utils import getAllFiles, getRootAndId


def update_distribute(type_distribute:dict, file_id:str):
    type = file_id.split("/")[0]
    if type not in type_distribute:
        type_distribute[type] = 0
    type_distribute[type] += 1
    return type_distribute


def processing_meta_data(data_root: str, save_meta_data: str):
    meta_data = {
        "图片绝对路径": list(),
        "图片相对路径": list(),
        "ocr结果": list(),
        "类型": list(),
        "用途": list()
    }

    type_distribute = dict()

    txt_files = getAllFiles(data_root, [".txt"])
    label_files = [file for file in txt_files if "Label.txt" in file]
    print(f"{len(label_files)}")
    for label_file in tqdm(label_files):
        with open(label_file, "r", encoding="utf-8") as f1:
            image_root, _ = getRootAndId(label_file)
            for sample in tqdm(f1):
                file_id, ocr_res = sample.strip().split("\t")
                absolute_image_path = os.path.join(image_root, file_id).replace("\\", "/")
                relative_image_path = absolute_image_path.split("/", 1)[1]
                meta_data["图片绝对路径"].append(absolute_image_path)
                meta_data["图片相对路径"].append(relative_image_path)
                meta_data["ocr结果"].append(ocr_res)
                meta_data["类型"].append(label_file.rsplit("/", 2)[1])
                meta_data["用途"].append(None)
                type_distribute = update_distribute(type_distribute, file_id)


    df = pd.DataFrame(meta_data)
    df.to_csv(save_meta_data, index=False)
    #df.to_excel(save_meta_data, index=False)

def processing_arrow_data(meta_data_file:str, output_path:str):
    dataset_features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "image": datasets.Value("string"),  # 路径
            "ocr_box": datasets.Value("string"),   # ocr
        }
    )
    writer = ArrowWriter(features=dataset_features, path=save_arrow_data)
    df = pd.read_csv(meta_data_file)
    print(df)
    for index, row in tqdm(df.iterrows()):
        #print(index, row)
        file_name = row["图片相对路径"]
        record = {
            "id": file_name,
            "image": row["图片绝对路径"],
            "ocr_box": row["ocr结果"]
        }
        example = dataset_features.encode_example(record)
        writer.write(example, file_name)
    writer.close()


if __name__ == "__main__":
    data_root = r"J:\data\mllm-data"
    save_meta_data = "J:\data\mllm-data\mllm-data-20231114.csv"
    save_arrow_data = "J:\data\mllm-data\mllm-data-20231114.arrow"
    gen_meta_data, gen_arrow_data = True, True
    if gen_meta_data:
        processing_meta_data(data_root, save_meta_data)

    if gen_arrow_data:
        processing_arrow_data(save_meta_data, save_arrow_data)
