# -*-coding: utf-8 -*-
import os
import copy

from .donut import convert_donut_predict
from .common import get_file_key

def covert_donut_birth_certifiacte(row):
    keys = [
        "新生儿姓名", "出生孕周", "出生体重", "母亲姓名", "母亲身份证号", "父亲姓名", "父亲身份证号", "出生证编号"
    ]
    view = convert_donut_predict(row, keys)
    view = create_detail_view(row)
    return view

def create_detail_view(row):
    """
    复杂的转换处理
    :param row:
    :return:
    """
    return row

def convert_donut_gt(row, keep_fail=True):
    if "predict" in row:
        return covert_donut_birth_certifiacte(row)
    if "抽取结果" in row:
        return convert_donut_gt(row)
    return copy.deepcopy(row)

def create_proj_view(row):
    return row

def create_bus_view(row):
    view = {k:v for k,v in row.items() if k in ["图片路径", "图片地址"] or k.startswith("__")}
    view["业务"] = [row.get(k, None) for k in ["新生儿姓名", "出生孕周", "出生体重"]]
    if all(i is None for i in view["业务"]):
        print(view["业务"])
    return view

def get_poc_file_key(x):
    return get_file_key(x, ["Z:/", "nlpdata/mllm_data/brith_certificate_data"])