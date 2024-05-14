"""
    测试ocr成行模块
"""
import time
import json

from mllm.utils.ocr_utils import PicInfo

"""
    测试ocr成行模块
"""

with open("./test_data/input_ocr_info.json", "r", encoding="utf-8") as fi:
    ocr_res = json.load(fi)
    t1 = time.time()
    pic_info = PicInfo.from_ocr_res(ocr_res)
    print(f"time:{time.time()-t1}")
    print(pic_info.rows)

