import os


# 方式一，使用os.walk(path)方法
def getAllFiles(path, suffix_list:list=None):
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(home, filename)
            file_suffix = os.path.splitext(file_path)[1]
            if suffix_list is not None and file_suffix.lower() in suffix_list:
                filelist.append(os.path.join(home, filename))
    return filelist


if __name__ == "__main__":
    path = r"J:\data\mllm-data"
    files = getAllFiles(path, [".jpg", ".png"])
    print(files)