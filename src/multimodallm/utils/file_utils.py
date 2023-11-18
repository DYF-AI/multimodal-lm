import os

from tqdm import tqdm


# 方式一，使用os.walk(path)方法
def getAllFiles(path, suffix_list:list=None):
    filelist = []
    for home, dirs, files in tqdm(os.walk(path)):
        for filename in tqdm(files):
            file_path = os.path.join(home, filename)
            file_suffix = os.path.splitext(file_path)[1]
            if suffix_list is not None and file_suffix.lower() in suffix_list:
                filelist.append(os.path.join(home, filename).replace("\\", "/"))
            elif suffix_list is None:
                filelist.append(os.path.join(home, filename).replace("\\", "/"))
    return filelist


# def getRootAndId(file_path:str):
#     file_path_split = file_path.split("/")
#     file_id = file_path_split[-2] + "/" + file_path_split[-1]
#     file_root = os.path.join(file_path_split[0], "/"+"/".join(file_path_split[1:-2]))
#     return file_root, file_id


if __name__ == "__main__":
    #path = r"J:\data\mllm-data"
    path = r"J:\data\mllm-data\image-book"
    files1 = getAllFiles(path, [".jpg", ".png"])
    files2 = getAllFiles(path)
    print(f"files1 nums:{len(files1)}")
    print(f"files2 nums:{len(files2)}")
