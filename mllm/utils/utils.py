import json
import re
from typing import Any, List, Dict


def filter_json_keys(data: Any, filter_keys: List[str]) -> Any:
    if isinstance(data, dict):
        return {k: filter_json_keys(v, filter_keys) for k, v in data.items() if k not in filter_keys}
    elif isinstance(data, list):
        return [filter_json_keys(item, filter_keys) for item in data]
    else:
        return data


def json2token(obj: Any, sort_json_key: bool = False, filter_keys: List[str] = None) -> str:
    # 如果有filter_keys，先过滤掉不需要的键
    if filter_keys:
        obj = filter_json_keys(obj, filter_keys)

    # 将 JSON 数据转换为字符串
    json_str = json.dumps(obj, ensure_ascii=False, sort_keys=sort_json_key)

    # 使用正则表达式去掉引号
    custom_format_str = re.sub(r'\"(\w+)\":', r'\1:', json_str)
    custom_format_str = re.sub(r'\"([^"]+)\"', r'\1', custom_format_str)

    custom_format_str = custom_format_str.replace(": ", ":").replace(", ", ",")

    return custom_format_str


import json


def token2json(token_str: str) -> Any:
    # 过滤不规则
    filter_str = [
        split_str for split_str in token_str.strip().split(",")
        if ":" in split_str
    ]
    filter_str = ",".join(filter_str)
    filter_str = filter_str.replace(":", ": ")

    # 使用正则表达式为键添加引号
    json_like_str = re.sub(r'(\w+):', r'"\1":', filter_str)

    # 为值添加引号
    # 首先处理不在括号内的值（简单的键值对）
    json_like_str = re.sub(r': ([^,\{\[\]\}]+)', r': "\1"', json_like_str)


    # 尝试将字符串转换为 JSON 对象
    try:
        json_data = json.loads(json_like_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e}")
        return None
    return json_data




if __name__ == '__main__':

    # 示例使用
    json_data = {
        "材料类型": "电子发票",
        "票据代码": "50068122",
        "票据号码": "8250626222",
        "校验码": "edfdf4",
        "交款人": "冯昱硕",
        "开票日期": "2023年03月16日",
        "费用明細": [
            {"细项名称": "卫生材料费", "金额": "8.98"},
            {"细项名称": "治疗费", "金额": "3.4"},
            {"细项名称": "化验费", "金额": "76.5"}
        ]
    }

    json_like_tokens = json2token(json_data, sort_json_key=True, filter_keys=["票据号码", "校验码"])
    print(f"json_like_tokens:{json_like_tokens}")
    parse_json = token2json(json_like_tokens)
    print(f"parse_json:{parse_json}, {parse_json['交款人']}")