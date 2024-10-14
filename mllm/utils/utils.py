import json
import re
from collections import defaultdict
from typing import Any, List, Dict


def filter_json_keys(data: Any, filter_keys: List[str]) -> Any:
    if isinstance(data, dict):
        return {k: filter_json_keys(v, filter_keys) for k, v in data.items() if k not in filter_keys}
    elif isinstance(data, list):
        return [filter_json_keys(item, filter_keys) for item in data]
    else:
        return data



def json2tokenV1(obj: Any, sort_json_key: bool = False, filter_keys: List[str] = None) -> str:
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


def json2tokenV2(obj:Any, sort_json_key: bool = False, prefix_list_of_dict: bool = False):
    """
    将复杂的数据结构flatten成kv的形式, 使用\n\n进行拼接, 要考虑顺序（python的dict还可以）
    :param concate_token:
    :param token_str:
    :param sort_json_key:
    :param obj:
    :return:
    """
    if sort_json_key and isinstance(obj, dict):
        obj = dict(sorted(obj.items(), key=lambda item: item[0]))

    result = []

    def parse_item(key, value, prefix=""):
        # 如果值是字典，递归处理
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parse_item(sub_key, sub_value, prefix=prefix)
        # 如果值是列表，处理列表中的每一项，并在每个字典内部字段用逗号拼接
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    sub_fields = []
                    for sub_key, sub_value in item.items():
                        full_key = f"{prefix}{key}-{sub_key}" if prefix_list_of_dict else sub_key
                        sub_fields.append(f"{full_key}:{sub_value}")
                    # 将字典内部字段拼接为字符串，结果列表分隔符为两行间隔
                    result.append(", ".join(sub_fields))
                elif isinstance(item, list):
                    # 递归处理嵌套的列表
                    parse_item(key, item, prefix=f"{prefix}{key}-" if prefix_list_of_dict else "")
                else:
                    # 处理列表中的基础类型项
                    result.append(f"{key}:{item}")
        # 基础类型直接添加到结果
        else:
            result.append(f"{prefix}{key}:{value}")

    # 遍历顶层键值对
    for key, value in obj.items():
        parse_item(key, value, prefix=f"" if prefix_list_of_dict and isinstance(value, list) else "")

    # 将所有内容合并，使用两行间隔
    return "\n\n".join(result)


def token2jsonV1(token_str: str) -> Any:
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


def token2jsonV2(token_str, prefix_list_of_dict=False):
    data = {}
    groups = []

    # 逐行处理输入字符串
    for line in token_str.strip().splitlines():
        line = line.strip()

        # 检查是否含有逗号
        if ',' in line:
            item = {}
            pairs = [pair.strip() for pair in line.split(',')]
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    item[key.strip()] = value.strip()
            groups.append(item)
        else:
            # 没有逗号的情况直接添加到 data 顶层结构
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

    # 根据 prefix_list_of_dict 参数，设置是否添加 "groups"
    if prefix_list_of_dict:
        # prefix_list_of_dict 为 True 时，将含有相同前缀的键值对分组
        structured_groups = defaultdict(list)
        for item in groups:
            # 提取键的前缀
            for key in item.keys():
                prefix = key.split('-')[0]
                structured_groups[prefix].append({k.split('-', 1)[-1]: v for k, v in item.items()})
                break
        data.update(structured_groups)
    else:
        # prefix_list_of_dict 为 False 时，直接加入 groups
        data["groups"] = groups

    return data

def mapping_dict_keys(obj:Any, key_mapping: Dict):
    if isinstance(obj, dict):
        return {
            key_mapping.get(k, k): mapping_dict_keys(v, key_mapping)
            for k, v in obj.items() if k in key_mapping
        }
    elif isinstance(obj, list):
        return [mapping_dict_keys(item, key_mapping) for item in obj]
    return obj



def demo_json2token_v1(json_data):
    # 示例使用
    json_like_tokens = json2tokenV1(json_data, sort_json_key=True, filter_keys=["票据号码", "校验码"])
    print(f"json_like_tokens:{json_like_tokens}")
    parse_json = token2jsonV1(json_like_tokens)
    print(f"parse_json:{parse_json}, {parse_json['交款人']}")

def demo_json2token_v2(json_data):
    # 示例使用
    token_str = json2tokenV2(json_data, sort_json_key=False, prefix_list_of_dict=False)
    print(f"json_to_tokens:{token_str}")
    result = token2jsonV2(token_str, prefix_list_of_dict=False)
    print(f"token_to_json：{result}")



if __name__ == '__main__':
    # 示例使用
    json_data = {
        "材料类型": "电子发票",
        "票据代码": "50068122",
        "票据号码": "8250626222",
        "校验码": "edfdf4",
        "交款人": "冯昱硕",
        "开票日期": "2023年03月16日",
        "费用明细": [
            {"细项名称": "卫生材料费", "金额": "8.98"},
            {"细项名称": "治疗费", "金额": "3.4"},
            {"细项名称": "化验费", "金额": "76.5"}
        ]
    }
    #demo_json2token_v1(json_data)
    demo_json2token_v2(json_data, )