"""
    测试ocr成行模块
"""
import time
import json

from multimodallm.ocr_system.ocr_info import PicInfo

with open("input_ocr_info.json", "r", encoding="utf-8") as fi:
    ocr_res = json.load(fi)
    t1 = time.time()
    pic_info = PicInfo.from_ocr_res(ocr_res)
    print(f"time:{time.time()-t1}")
    print(pic_info.rows)

