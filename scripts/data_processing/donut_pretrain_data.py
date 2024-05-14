
"""
    将donut的pretrain数据,进行整理,在数据根目录生成一个metadata.json
"""
import json
import os

from tqdm import tqdm

from mllm.utils.ocr_utils import PicInfo
from mllm.utils.data_utils import ocr_rows_text

def adapte_ocr_data(ocr_res, file_id):
    ocr_res_data = json.loads(ocr_res)
    adapter_data = {
        "file_id": file_id,
        "bboxes": ocr_res_data,
    }
    return adapter_data

def get_label_files(data_root:str):
    usages = ["train", "validation", "test"]
    label_files = []
    for usage in tqdm(usages):
        usage_root = os.path.join(data_root, usage)
        if not os.path.exists(usage):
            continue
        for type in os.listdir(usage_root):
            type_root = os.path.join(usage_root, type)
            label_files.append(os.path.join(type_root, "Label.txt"))
    return label_files

def cal_ocr_score(boxes):
    return sum([box.bbox_score for box in boxes])/(len(boxes)+0.0005)

def processing_meta_data(data_root: str, save_meta_data: str):

    with open(save_meta_data, "w", encoding="utf-8") as fo:
        label_files = get_label_files(data_root)
        print(f"{len(label_files)}")
        for label_file in tqdm(label_files):
            label_file = label_file.replace("\\", "/")
            print(f"label_file:{label_file}")
            with open(label_file, "r", encoding="utf-8") as f1:
                image_root = label_file.rsplit("/", 2)[0]
                label_file_split = label_file.split("/")
                usage, type = label_file_split[4], label_file_split[5]
                for sample in tqdm(f1):
                    file_id, ocr_res = sample.strip().split("\t")
                    absolute_image_path = os.path.join(image_root, file_id).replace("\\", "/")
                    relative_image_path = absolute_image_path.split("/", 3)[3]
                    if not os.path.isfile(absolute_image_path):
                        continue
                    picInfo = PicInfo.from_ocr_res(adapte_ocr_data(ocr_res, file_id))
                    row_data = {
                        "数据来源": "DYF",
                        "数据用途": usage,
                        "图片路径": absolute_image_path.replace(data_root, ""),
                        "抽取结果": "",
                        "OCR文本": ocr_rows_text(picInfo.rows),
                        "OCR坐标": ocr_res,
                        "OCR分数": cal_ocr_score(picInfo.bounding_boxes),
                        "图片角度": 0.0,
                        #"DATA_ROOT": data_root
                    }
                    fo.write(json.dumps(row_data, ensure_ascii=False) + "\n")



if __name__ == '__main__':
    DATA_ROOT = "M:/datasets/mllm-data/mllm-pretrain-data/"
    SAVE_FILE = "M:/dataset/mllm-data/mllm-pretrain-data/metadata.jsonl"

    processing_meta_data(DATA_ROOT, SAVE_FILE)

