import os
import shutil
from tqdm import tqdm

def rename(image_root:str, feature="%", new_name_startwith="image")->bool:
    files = [file for file in os.listdir(image_root) if file.startswith(feature)]
    for index, file_name in enumerate(tqdm(files)):
        try:
            src_file = os.path.join(image_root, file_name)
            file_split = os.path.splitext(file_name)
            new_file_name = new_name_startwith + f"_{index+100}" + file_split[1]
            dst_file = os.path.join(image_root, new_file_name)
            shutil.move(src_file, dst_file)
            #print(src_file ,dst_file)
        except Exception as e:
            print(f"P{file_name}, {e}")
            #return False
    return True

if __name__ == "__main__":
    flag = rename(r"J:\data\mllm-data\crawler-data\image\判决书")