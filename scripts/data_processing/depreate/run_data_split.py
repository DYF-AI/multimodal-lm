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
import os.path
import shutil
import random

from tqdm import tqdm
from mllm.utils.file_utils import getAllFiles



def split_and_move_data(data_root:str):
    # 1.在data_root下找到所有的label文件
    txt_files = getAllFiles(data_root, [".txt"])
    label_files = [file for file in txt_files if "Label.txt" in file]

    filter_files = []
    for file in label_files:
        new_file_name = file.replace("pretrain", "")
        if ("train" not in new_file_name) and ("validation" not in new_file_name) and ("test" not in new_file_name):
            filter_files.append(file)
    label_files = filter_files

    split = {
        "book": (0.95, 0.025, 0.025),
        "other": (0.90, 0.05, 0.05)
    }
    train_root = os.path.join(data_root, "train")
    validation_root = os.path.join(data_root, "validation")
    test_root = os.path.join(data_root, "test")
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(validation_root):
        os.makedirs(validation_root)
    if not os.path.exists(test_root):
        os.makedirs(test_root)

    error_files = []
    for label_file in tqdm(label_files):
        all_files = list()
        with open(label_file, "r", encoding="utf-8") as f1:
            for line in tqdm(f1):
                 all_files.append(line)
        random.shuffle(all_files)
        if "book" in label_file:
            proportion = split["book"]
        else:
            proportion = split["other"]

        a, b = int(len(all_files)*proportion[1]), int(len(all_files)*proportion[1])+int(len(all_files)*proportion[2])
        validation_files = all_files[:a]
        test_files = all_files[a:b]
        train_files = all_files[b:]
        print(len(train_files), len(validation_files), len(test_files))

        data = {
            "validation": validation_files,
            "test": test_files,
            "train": train_files
        }

        for key in tqdm(data.keys()):
            split_files = data[key]
            dst_root = os.path.join(data_root, key)
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)
            file_id, _ = split_files[0].split("\t")
            new_folder = file_id.split("/")[0]
            dst_image_root = os.path.join(dst_root, new_folder)
            if not os.path.exists(dst_image_root):
                os.makedirs(dst_image_root)
            new_label_file = os.path.join(dst_image_root, "Label.txt")
            new_state_file = os.path.join(dst_image_root, "fileState.txt")
            with open(new_label_file, "w", encoding="utf-8") as f2, open(new_state_file, "w", encoding="utf-8") as f3:
                for sample in tqdm(split_files):
                    file_id, ocr_res = sample.strip().split("\t")
                    src_image_path = os.path.join(data_root, file_id)#.replace("\\", "/")
                    # filestate路径貌似必须这么个格式
                    dst_image_path = os.path.join(dst_root, file_id).replace("/", "\\")
                    try:
                        shutil.copy(src_image_path, dst_image_path)
                    except Exception as e:
                        print(f"{src_image_path} copy error {e}, continue!")
                        error_files.append(src_image_path)
                        print(f"error num: {len(error_files)}")
                        continue
                    print(src_image_path, dst_image_path)
                    f2.write(file_id + "\t" + ocr_res + "\n")
                    f3.write(dst_image_path+ "\t" + str(1) +"\n")
                    f2.flush()
                    f3.flush()


            # validation
            #file_root, _ = os.path.split(label_file)
            # new_label_file = os.path.join(file_root, "Label.txt")
            # new_state_file = os.path.join(file_root, "fileState.txt")
            # with open(new_label_file, "w", encoding="utf-8") as f2, open(new_state_file, "w", encoding="utf-8") as f3:
            #     for validation_file in validation_files:
            #         file_id, ocr_res = validation_file.split("\t")
            #         src_image_path = os.path.join(data_root, file_id).replace("\\", "/")
            #         dst_image_path = os.path.join(validation_root, file_id).replace("\\", "/")
            #         new_file_root, _ = os.path.split(dst_image_path)
            #         if not os.path.exists(new_file_root):
            #             os.makedirs(new_file_root)
            #         shutil.copy(src_image_path, dst_image_path)
            #         print(src_image_path, dst_image_path)
            #         f2.write(validation_file + "\n")
            #         f3.write(dst_image_path+ "\t" + str(1) +"\n")
            #         f2.flush()
            #         f3.flush()

    print(f"error_files: {error_files}  {len(error_files)}")


if __name__ == "__main__":
    data_root = "J:/data/mllm-data/mllm-pretrain-data"
    split_and_move_data(data_root)



