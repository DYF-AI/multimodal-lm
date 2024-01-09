# -*- coding:ytf-8 -*-

import os
import json
import logging
import pandas as pd
from .util import str_q2b

logger = logging.getLogger("adapter")


def convert_ocr_to_image(pic_id_or_pic_id_list, ocr_to_image=None):
    if not ocr_to_image:
        return None
    if not isinstance(pic_id_or_pic_id_list, list):
        pic_id = pic_id_or_pic_id_list
        if callable(ocr_to_image):
            return ocr_to_image(pic_id)
        else:
            if pic_id not in ocr_to_image:
                logger.warning(f"{pic_id} is not found")
                return None
            return ocr_to_image[pic_id]
    pic_id_list = pic_id_or_pic_id_list
    if callable(ocr_to_image):
        pic_list = [ocr_to_image(v) for v in pic_id_list]
    else:
        unknown_pic_list = [v for v in pic_id_list if v not in ocr_to_image]
        if unknown_pic_list:
            logger.warning(f"{unknown_pic_list} is not found")
        pic_list = [ocr_to_image(v) for v in pic_id_list if v in ocr_to_image]
    return pic_list


def add_image_in_row(row, ocr_to_image):
    if isinstance(row, list):
        return [add_image_in_row(v, ocr_to_image) for v in row]
    if not isinstance(row, dict) or "value" in row:
        return row
    extra_info = dict()
    for k, v in row.items():
        if isinstance(v, dict):
            add_image_in_row(v, ocr_to_image)
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for vv in v:
                add_image_in_row(vv, ocr_to_image)
        is_ocr = False
        if k in ["ocr文件路径", "ocr列表", "ocr文件夹路径"]:
            is_ocr = True
        if not is_ocr:
            if isinstance(v, str) and v.endswith(".txt"):
                is_ocr = True
            if isinstance(v, list) and all([isinstance(vv, str) and vv.endswith(".txt") for vv in v]):
                is_ocr = True
        if is_ocr:
            newk = k.replace("ocr", "图片")
            image_info = convert_ocr_to_image(v, ocr_to_image)
            if image_info is not None:
                extra_info[newk] = image_info
    if extra_info:
        row.update(extra_info)
    return row


def convert_file(infile, convert_fn, ocr_to_image=None, save_path=None, use_cache=None, key=None, return_dict=False,
                 keep_empty=False):
    if return_dict and key is None:
        raise ValueError("key not provided")

    if use_cache and save_path is not None and os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as fi:
            return json.load(fi)

    if infile.endswith(".json"):
        data_list = load_json_file(infile)
    elif infile.endswith(".jsonl"):
        data_list = load_jsonl_file(infile)
    else:
        raise ValueError(f"unsupported file {infile}")
    resutl = dict() if return_dict else list()
    for row in data_list:
        data = convert_fn(row)
        if data is None:
            continue
        data = add_image_in_row(data, ocr_to_image)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and len(data) == 0 and keep_empty:
            data = [{}]
        for item in data:
            if key is not None:
                rowid = None
                if isinstance(key, str):
                    if key in item:
                        if not item[key]:
                            logger.warning("ignore becaues empty id")
                            continue
                        rowid = item[key]
                elif callable(key):
                    rowid = key[item]
                    if isinstance(rowid, tuple):
                        rowid, keyname = rowid
                        item[keyname] = rowid
                    else:
                        item["__id__"] = rowid
                if return_dict:
                    resutl[rowid] = item
                else:
                    resutl.append(item)
            else:
                resutl.append(item)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fo:
            json.dump(resutl, fo, ensure_ascii=False, indent=2)
    return resutl

def load_json_file(infile):
    with open(infile, encoding="utf-8") as fi:
        content = str_q2b(fi.read())
    data_list = json.loads(content)
    return data_list


def load_jsonl_file(infile):
    with open(infile, encoding="utf-8") as fi:
        line = next(fi)
        try:
            json.loads(line)
            is_json = True
        except:
            is_json = False
    if is_json:
        data_list = list()
        with open(infile, encoding="utf-8") as fi:
            for line in fi:
                row = json.loads(str_q2b(line))
                data_list.append(row)
    else:
        data_list = list()
        with open(infile, encoding="utf-8") as fi:
            for line in fi:
                filepath = line.strip().strip().replace("\\", "/")
                row = json.loads(str_q2b(next(fi)))
                if filepath.endswith(".txt"):
                    row["ocr文件路径"] = filepath
                if filepath.endswith(".jpg") or filepath.endswith(".png") or filepath.endswith(".jpeg"):
                    row["图片路径"] = filepath
                else:
                    row["ocr文件夹路径"] = filepath
                data_list.append(row)
    return data_list


def get_file_key(x, prefix=None, remove_ext=False, remove_split_id=False):
    if isinstance(x, str):
        path = x
    else:
        if "图片文件路径" not in x and "图片路径" not in x and "ocr文件路径" not in x:
            return None
        if "图片文件路径" in x:
            path = x["图片文件路径"]
        elif "图片路径" in x:
            path = x["图片路径"]
        else:
            path = x["ocr文件路径"]

    if not path:
        return None

    path = path.replace("\\", "/")
    if prefix is None and remove_split_id is False:
        return path

    dirpath = os.path.dirname(path)
    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            dirpath = dirpath.replace(prefix, "")

    filename = os.path.basename(path)
    if remove_split_id:
        name, ext = os.path.splitext(filename)[0]
        name = name.split("_")[0]
        filename = f"{name}-{ext}"
    if remove_ext:
        filename = os.path.splitext(filename)[0]
    return f"{dirpath}/{filename}"

def load_id(id_file):
    id_list = list()
    if id_file.endswith(".txt"):
        with open(id_file, encoding="utf-8") as fi:
            for line in fi:
                id_list.append(line.strip())
    elif id_file.endswith(".xlsx"):
        df = pd.read_excel(id_file)
        id_list = df["图片路径"].tolist()
    return id_list











































































































