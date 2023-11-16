import os
import shutil
from tqdm import tqdm

"""
    只要针对cv2无法进行读取的文件名进行修改, 使用魔改的ppocrlabel-pil可以解决文件路径读取的问题
"""
def rename(image_root:str, feature="%", new_name_startwith="image")->bool:
    files = [file for file in os.listdir(image_root) if file.startswith(feature)]
    for index, file_name in enumerate(tqdm(files)):
        try:
            src_file = os.path.join(image_root, file_name).replace("\\", "/")
            file_split = os.path.splitext(file_name)
            new_file_name = new_name_startwith + f"_{index+100}" + file_split[1]
            dst_file = os.path.join(image_root, new_file_name).replace("\\", "/")
            shutil.move(src_file, dst_file)
            #print(src_file ,dst_file)
        except Exception as e:
            print(f"P{file_name}, {e}")
            #return False
    return True

if __name__ == "__main__":
    flag = rename(r"J:\data\mllm-data\mllm-pretrain-data\crawler-data-法律法规")