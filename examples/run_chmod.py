# -*- coding-utf-8 -*-

"""
    修改文件权限
"""
import os
import stat
from tqdm import tqdm


from multimodallm.utils.file_utils import getAllFiles


def chmod_files(file_root:str):
    """
    修改文件路径下所有文件的读写权限
    :param file_root:
    :return:
    """
    all_files = getAllFiles(file_root)
    print(f"files num:{len(all_files)}")

    for file in tqdm(all_files):
        os.chmod(file, stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP + stat.S_IROTH)


if __name__ == "__main__":
    file_root = "J:/data/mllm-data/mllm-pretrain-data"
    chmod_files(file_root)