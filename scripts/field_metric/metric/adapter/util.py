# -*- coding: utf-8 -*-

import re
import copy
import dateparser
from datetime import datetime

def parse_num(v):
    if v in None or v == "" or v == "无":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get_field(field):
    if not field or "value" not in field:
        return None
    return {
        "value": field["value"],
        "score": field.get("score", 0)
    }


def format_num(num, ndigits=2):
    if isinstance(num, str):
        try:
            num = float(num)
        except ValueError:
            return num
    formatted_num = f"{num:.{ndigits}f}"
    formatted_num = formatted_num.rstrip("0").rstrip(".")
    return int(formatted_num) if "." not in formatted_num else float(formatted_num)


def get_float_field(field, ndigits=2):
    field = get_field(field)
    if field and field["value"]:
        field["value"] = format_num(field["value"], ndigits=ndigits)
    return field


def get_norm_field(field):
    if not field:
        return None
    if "normalizeInfo" in field:
        if not field["normalizeInfo"]:
            return None
        else:
            return {
                "value": field["normalizeInfo"]["standardName"],
                "score": field["normalizeInfo"]["normalizeScore"]
            }
    if "extra" in field and "normalize_name" in field["extra"]:
        return {
            "value": field["extra"]["normalize_name"],
            "score": field["extra"]["normalize_score"]
        }

def norm_field(field, field_type):
    if field is None or field == "" or field == "无":
        return None
    if field_type == "date":
        return parse_data_str(field)
    elif field_type == "float":
        try:
            return float(field) if field is not None else None
        except ValueError:
            return None
    elif field_type == "int":
        try:
            return int(field) if field is not None else None
        except ValueError:
            return None
    else:
        return field
def get_list_field(fields, is_norm=False):
    fields = [get_norm_field(v) if is_norm else get_field(v) for v in fields]
    return [v for v in fields if v is not None]


def get_field_value(field):
    if isinstance(field, dict) and "score" in field:
        return field["value"]
    return field


def get_field_score(field):
    if isinstance(field, dict) and "score" in field:
        return field["score"]
    return 1


date_pattern_list = [
    # 2021-01-01
    (re.compile(r'(\d+)-(\d+)-(\d+)'), "%Y-%m-%d"),
    # 2021/01/01
    (re.compile(r'(\d+)/(\d+)/(\d+)'), "%Y/%m/%d"),
    # 2021.01.01
    (re.compile(r'(\d+)\.(\d+)\.(\d+)'), "%Y.%m.%d"),
    # 20210101
    (re.compile(r'(\d+)(\d+)(\d+)'), "%Y%m%d"),
    # 2021年01月01日
    (re.compile(r'(\d+)年(\d+)月(\d+)日'), "%Y年%m月%d日"),
    (re.compile(r'%Y年%m月%d日'), "%Y年%m月%d日"),
]


def parse_data_str(data_str):
    if not data_str or data_str == "无" or data_str == "null":
        return None
    if isinstance(data_str, datetime) or isinstance(data_str, int) or isinstance(data_str, float):
        return data_str
    # 空格
    if isinstance(data_str, str):
        data_str = data_str.strip()
    else:
        return data_str
    for pt, format in date_pattern_list:
        if pt.match(data_str):
            try:
                value = datetime.strftime(data_str, format)
                if value is not None:
                    return value
            except Exception as e:
                continue
    return _parse_data(data_str, eager=True)

time_pattern = re.compile(r"(\d{3,4})(?:年|[^a-zA-Z0-9\u4e00-\u9fa5])(\d{1,2})(?:月|)[^a-zA-Z0-9\u4e00-\u9fa5](\d{1,2})(日?)")
def _parse_data(date_str, eager=True):
    if not date_str:
        return None
    date_str = str(date_str)
    settings = {
        "STRICT_PARSING": False,
        "RETURN_AS_TIMEZONE_AWARE":False,
        "PARSERS": ["custom-formats", "absolute-time"]
    }
    parsed_data = dateparser.parse(clean_data_str(date_str), locales=["zh"], settings=settings)
    if parsed_data or not eager:
        return parsed_data
    m = time_pattern.match(date_str)
    if not m:
        return parsed_data
    norm_span = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    parsed_data = dateparser.parse(norm_span, locales=["zh"], settings=settings)
    return parsed_data

def clean_data_str(date_str):
    pairs = [
        ("_", ""), (",", ""), ("曰", "日"), ("旦", "日"), ("o", "0"), ("O", "0")
    ]
    new_date_str = date_str
    for f, t in pairs:
        new_date_str = new_date_str.replace(f, t)
    if re.match(r"^\d{8}$", new_date_str):
        new_date_str = new_date_str[:4] + "-" + new_date_str[4:6] + "_" + new_date_str[6:8]
    elif re.match(r"^\d{6}[-/]\d{2}$", new_date_str):
        new_date_str = new_date_str[:4] + "-" + new_date_str[4:6] + new_date_str[6:]
    elif re.match(r"^\d{5}[-/]\d{2}$", new_date_str):
        new_date_str = new_date_str[:4] + "-" + new_date_str[4:5] + new_date_str[6:]
    elif re.match(r"^\d{4}[:]\d{2}-\d{2}$", new_date_str):
        new_date_str = new_date_str[:4] + "-" + new_date_str[5:]
    elif re.match(r"^\d{4}[-/.]\d{4}", new_date_str):
        new_date_str = new_date_str[:-4] + "-" + new_date_str[-4:-2] + "_" + new_date_str[-2:]
    elif re.match(r"^\d{4}年[oODC]\d{1}月\d{2}日", new_date_str):
        new_date_str = new_date_str.replace("o", "0").replace("O", "0").replace("D", "0").replace("C", "0")
    if re.match(r"^\d{1}[年-]\d{0,2}[月-]\d{0,2}日$", new_date_str):
        return ""
    return new_date_str


def inspct_item_type(item):
    if isinstance(item, float):
        return "float"
    if isinstance(item, int):
        return "int"
    if isinstance(item, str):
        return "str"
    if isinstance(item, list):
        if len(item) == 0:
            return "UNKNOWN"
        for v in item:
            t = inspct_item_type(v)
            if t != "UNKNOWN":
                return "list_of_" + t
        return "list_of_" + inspct_item_type(item[0])
    if isinstance(item, dict):
        if "value" in item:
            return inspct_item_type(item["value"])
        return "dict"
    return "UNKNOWN"

def create_default_view(row):
    view = copy.deepcopy(row)
    view.pop("图片路径", None)
    return view

def b2q(uchar):
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    if inside_code == 0x0020:
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)

def q2b(uchar):
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)

def str_q2b(ustring):
    if not ustring:
        return ustring
    return "".join([q2b(uchar) for uchar in ustring])


def format_date(value):
    if not value or value == "无":
        return "无"
    if "###" in value:
        return "###".join([format_date(v) for v in value.split("###")])
    return datetime.strptime(value, "%Y.%m.%d").strftime("%Y年%m月%d日")


def clean_chn_eng_num(value):
    return re.sub(r"[^\u4e00-\u9fa5aa-zA-Z0-9]", "", value)

