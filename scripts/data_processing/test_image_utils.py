import os.path
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
from multimodallm.utils.image_utils import load_image, pdf_to_images
from multimodallm.utils.file_utils import getAllFiles

"""
    测试读取图片
"""
def test_load_image():
    #image_path = "/mnt/j/data/mllm-data/image/病历资料/0b7b02087bf40ad1d7089f2a552c11dfa8ecce0f.jpg"
    #image_path = "J:/data/mllm-data/image/病历资料/0b7b02087bf40ad1d7089f2a552c11dfa8ecce0f.jpg"
    image_path = "J:/data/mllm-data/image/保单材料/u=2826187794,2771569356&fm=26&fmt=auto&gp=0.jpg"
    image_data, (w,h) = load_image(image_path, return_chw=False, size=(1280,960))
    print(image_data, (w, h))



if __name__ == "__main__":
    test_load_image()

