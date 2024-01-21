# -*- coding: utf-8 -*-

import os
import re
import copy
import yaml
import json
import pandas as pd

from .adapter.util import get_field_value,\
    get_field_score, norm_field, inspct_item_type, create_default_view, clean_data_str, clean_chn_eng_num
from .util import string_similar, sort_and_union_values
from .common import logger
from tqdm import tqdm

PRED_TOTAL_KEY, PRED_RIGHT_KEY = "预测数量", "预测准确数量"
RECALL_TOTAL_KEY, RECALL_RIGHT_KEY = "GT数量", "召回准确数量"

def is_field_value_empty(value):
    if value is None or value == "" or value == "无" or value == []:
        return True
    return False

def match_atom_field_score(left_field, right_field, field_type="str", field_empty_ok=False, **kwargs):
    left_field = get_field_value(left_field)
    right_field = get_field_value(right_field)
    left_field = norm_field(left_field, field_type)
    right_field = norm_field(right_field, field_type)

    if is_field_value_empty(left_field):
        if field_empty_ok:
            return 1 if is_field_value_empty(right_field) else 0
        return 0

    if field_type in ["date", "float", "int"]:
        return 1. if left_field == right_field else 0.
    elif field_type == "str":
        left_field_list = left_field.split("///") if left_field is not None else []
        right_field_list = right_field.split("///") if right_field is not None else []
        if kwargs.get("match_type", "") == "chn+eng+num":
            left_field_list = [clean_chn_eng_num(v) for v in left_field_list]
            right_field_list = [clean_chn_eng_num(v) for v in right_field_list]
        similarity = kwargs.get("similarity", 1)
        if similarity == 1 or not similarity:
            eq = 0
            for v1 in left_field_list:
                for v2 in right_field_list:
                    if v1 == v2:
                        eq += 1
            score = 1. if eq > 0 else 0.
        else:
            max_score = 0
            for v1 in left_field_list:
                for v2 in right_field_list:
                    max_score = max(string_similar(v1, v2), max_score)
            return max_score
        return score
    else:
        raise ValueError(f"not supported type {field_type}")

def match_atom_field(left_field, right_field, key=None, field_type="str", threshold=0., field_empty_ok=False, **kwargs):
    left_field_score = get_field_score(left_field)
    right_field_score = get_field_score(right_field)
    if left_field_score < threshold:
        return {
            "key": key,
            "right": 0,
            "total": 0,
            "detail_list":[]
        }
    left_field_value = get_field_value(left_field)
    right_field_value = get_field_value(right_field)

    norm_left_field_value = norm_field(left_field_value, field_type)
    if is_field_value_empty(norm_left_field_value):
        total = 1 if field_empty_ok else 0
    else:
        total = 1
    if right_field_score < threshold:
        return {
            "key": key,
            "right": 0,
            "total": total,
            "detail_list": [{"key":key, "left_field": left_field, "right_field":None, "is_right":0}]
        }
    score = match_atom_field_score(left_field_value, right_field_value, field_type, field_empty_ok, **kwargs)
    if field_type == "str":
        similarity = kwargs.get("similarity", 1)
        if similarity and similarity != 1:
            right = 1 if score >= similarity else 0
        else:
            right = 1 if score == 1 else 0
    else:
        right = 1 if score == 1 else 0
    return {
        "key": key,
        "right": right,
        "total": total,
        "detail_list": [{"key": key, "left_field": left_field, "right_field": right_field, "is_right": right}]
    }

def get_most_match_index(left_field, right_field_list, main_key, field_type="str", **kwargs):
    if callable(main_key):
        return main_key(left_field, right_field_list)
    if main_key not in left_field or right_field_list is None:
        return -1, 0

    max_match_index, max_match_score = -1 ,0
    left_value = left_field[main_key]
    min_score = kwargs.get("similarity", 0)
    for i, right_field in enumerate(right_field_list):
        if main_key not in right_field:
            continue
        right_value = right_field[main_key]
        score = match_atom_field_score(left_value, right_value, field_type=field_type, **kwargs)
        if score < min_score:
            continue
        if score > max_match_score:
            max_match_index = 1
            max_match_score = score
    return max_match_index, max_match_score

def get_match_score_list(left_field_list, right_field_list, main_key, field_type="str", **kwargs):
    match_score_list = list()
    for i, left_field in enumerate(left_field_list):
        for j, right_field in enumerate(right_field_list):
            if main_key not in left_field or main_key not in right_field:
                match_score_list.append((i, j, 0))
                continue
            score = match_atom_field_score(left_field[main_key], right_field[main_key], field_type=field_type, **kwargs)
            match_score_list.append((i, j, score))
    return match_score_list


black_key_list = [
    re.compile("__*"), re.compile(".*路径.*"), re.compile(".ocr.*"), re.compile(".*图[像片].*"),
    re.compile(".*文件夹.*"), "案件号", "用途", "标注文件", "未提及"
]


def match_dict_field(left_field, right_field, prefix=None, match_kwargs=None):
    if right_field is None:
        right_field = dict()
    if match_kwargs is None:
        match_kwargs = dict()
    threshold = match_kwargs.get("threshold", 0.)
    field_empty_ok =match_kwargs.get("field_empty_ok", False)

    stat_info = dict()
    for k, v in left_field.items():
        ignore = False
        for pt in black_key_list:
            if isinstance(pt, str) and pt == k:
                ignore = True
                break
            if isinstance(pt, re.Pattern) and pt.match(k):
                ignore = True
                break
            if k == "分组" and isinstance(v, str):
                ignore = True
                break
        if ignore:
            continue

        full_key = f"{prefix}-{k}" if prefix else k
        cur_match_kwargs = copy.deepcopy(match_kwargs.get(k, dict()))
        cur_match_kwargs = get_match_config(cur_match_kwargs, threshold=threshold, field_empty_ok=field_empty_ok)
        match_list = match_field_adapter(left_field.get(k), right_field.get(k), key=full_key, match_config=match_kwargs)
        for match_info in match_list:
            match_key = match_info["key"]
            if match_key not in stat_info:
                stat_info[match_key] = {"key":match_key, "right":0, "total":0, "detail_list":list()}
            stat_info[match_key]["total"] += match_info["total"]
            stat_info[match_key]["right"] += match_info["right"]
            stat_info[match_key]["detail_list"] += match_info["detail_list"]
    return list(stat_info.values())
def get_match_config(match_config, default_type=None, threshold=0, field_empty_ok=False):
    if match_config is False:
        return match_config
    if isinstance(match_config, str) and match_config in {"str", "int", "float", "date"}:
        return {"field_type": match_config, "threshold": threshold, "field_empty_ok": field_empty_ok}
    if match_config:
        if "similarity" in match_config and "field_type" not in match_config:
            match_config["field_type"] = "str"
        if "threshold" not in match_config:
            match_config["threshold"] = threshold
        if "field_type" not in match_config and default_type:
            match_config["field_type"] = default_type
        if "field_empty_ok" not in match_config:
            match_config["field_empty_ok"] = field_empty_ok
        return match_config
    if default_type is not None:
        return {"field_type": default_type, "threshold": threshold, "field_empty_ok": field_empty_ok}
    return {"threshold": threshold, "filed_empty_ok": field_empty_ok}

def match_field_adapter(left_item, right_item, match_config, key=""):
    if match_config is False:
        return []
    threshold = match_config.get("threshold", 0.) if isinstance(match_config, dict) else 0.
    field_empty_ok = match_config.get("field_empty_ok", False) if isinstance(match_config, dict) else False
    key_name = match_config["__name__"] if isinstance(match_config, dict) and "__name__" in match_config else False
    left_type = inspct_item_type(left_item)
    if "UNKOWN" == left_type:
        return []

    if is_field_value_empty(left_item):
        kwargs = get_match_config(match_config, default_type="str", threshold=threshold, field_empty_ok=field_empty_ok)
        match_info = match_atom_field(left_item, right_item, key=key, **kwargs)
        return [match_info]

    if left_type in ["int", "float", "str", "date"]:
        kwargs = get_match_config(match_config, default_type=left_type, threshold=threshold, field_empty_ok=field_empty_ok)
        match_info = match_atom_field(left_item, right_item, key=key, **kwargs)
        return [match_info]

    if left_type == "dict":
        match_kwargs = get_match_config(match_config, default_type=left_type, threshold=threshold,
                                  field_empty_ok=field_empty_ok)
        match_list = match_dict_field(left_item, right_item, prefix=key_name, match_kwargs=match_kwargs)
        return match_list

    if left_type == "list_of _dict":
        if match_config is None:
            match_config = dict()
        match_by = match_config.get("__match_by__")
        match_kwargs = get_match_config(match_config, threshold=threshold, field_empty_ok=field_empty_ok)
        match_list = match_list_of_dict_field(left_item, right_item, prefix=key_name,
                                              match_by=match_by, match_kwargs=match_kwargs)
        return match_list

    if left_type in {"list_of_int", "list_of_float", "list_of_str"}:
        kwargs = get_match_config(match_config, default_type=left_type.replace("list_of_", ""), threshold=threshold,
                                  field_empty_ok=field_empty_ok)
        match_info = match_list_of_atom_field(left_item, right_item, key=key_name, **kwargs)
        return [match_info]

    return []

def match_list_of_dict_field(left_field_list, right_field_list, prefix=None, match_by=None, match_kwargs=None):


    return []

def match_list_of_atom_field(left_field_list, right_field_list, key=None, all_right=False, field_type="str", **kwargs):

    return []