import json
import os
import pickle

from tqdm import tqdm


def get_label_files(data_root:str):
    usages = ["train", "validation", "test"]
    label_files = []
    for usage in tqdm(usages):
        usage_root = os.path.join(data_root, usage)
        label_files.append(os.path.join(usage_root, "metadata.jsonl"))
    return label_files

usage_mapping = {
    "test" : "测试",
    "validation": "验证",
    "train": "训练"
}


def processing_meta_data(data_root: str, save_meta_data: str):
    ori_metadata_files = get_label_files(data_root)
    with open(save_meta_data, "w", encoding="utf-8") as fo:
        for ori_metadata_file in ori_metadata_files:
            ori_metadata_file = ori_metadata_file.replace("\\", "/")
            if not os.path.exists(ori_metadata_file):
                print(f"{ori_metadata_file} not exits, continue!")
                continue
            with open(ori_metadata_file, "r", encoding="utf-8") as fi:
                ori_metadata_file = ori_metadata_file.replace(data_root, "")
                usage = ori_metadata_file.split("/")[0]
                for line in fi:
                    ori_row_data = json.loads(line)
                    new_row_data = {
                        "数据来源": "EATEN",
                        "数据用途": usage_mapping[usage],
                        "图片路径": f"{usage}/{ori_row_data['file_name']}",
                        "抽取结果": ori_row_data["text"],
                        "OCR文本": "",
                        "OCR坐标": "",
                        "OCR分数": "",
                        "图片角度": "",
                        "DATA_ROOT": data_root
                    }
                    fo.write(json.dumps(new_row_data, ensure_ascii=False) + "\n")




if __name__ == '__main__':
    DATA_ROOT = "N:/data/mllm-data/mllm-finetune-data/trainticket"
    SAVE_FILE = "N:/data/mllm-data/mllm-finetune-data/trainticket/metadata.jsonl"

    processing_meta_data(DATA_ROOT, SAVE_FILE)