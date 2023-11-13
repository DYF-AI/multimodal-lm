import os
import cv2
import shutil
import numpy as np
from multimodallm.utils.image_utils import load_image
from PyQt5.QtGui import QImage, QCursor, QPixmap, QImageReader

"""
    测试cv2读取图片错误
"""

image_root = r"J:\data\mllm-data\image-crawler"

dirs = [dir for dir in os.listdir(image_root) if not dir.endswith(".txt")]

fail_num, total_num = 0, 0
for dir in dirs:
    dir_path = os.path.join(image_root, dir)
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if not os.path.isfile(file_path) or file_path.endswith(".cach") or file_path.endswith(".txt"):
            continue
        try:
            use_cv = False
            if use_cv:
                cvimg = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
                height, width, depth = cvimg.shape
                cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                image = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
            else:
                cvimg1, _ = load_image(file_path, return_chw=False)
                height, width, depth = cvimg1.shape
                cvimg1 = cv2.cvtColor(cvimg1, cv2.COLOR_BGR2RGB)
                image1 = QImage(cvimg1.data, width, height, width * depth, QImage.Format_RGB888)


        except Exception as e:
            print(f"{e} load fail, remove")
            #os.remove(file_path)
            fail_num+=1
        total_num += 1
        print(fail_num, total_num)
        print(file_path)
        if not file_path.endswith(".gif"):
            continue
        new_file_path = file_path+".jpg"
        print(file_path, new_file_path)
        #shutil.move(file_path, new_file_path)
print(f"fail_num:{fail_num}")