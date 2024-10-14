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


def json2tokenV2(data, sort_json_key=False, prefix_list_of_dict=True):
    if sort_json_key and isinstance(data, list):
        data = dict(sorted(data.items(), key=lambda item: item[0]))
    def parse_dict(d, prefix=''):
        tokens = []
        for k, v in d.items():
            new_prefix = f"{prefix}-{k}" if prefix else k
            if isinstance(v, dict):
                tokens.extend(parse_dict(v, new_prefix))
            elif isinstance(v, list):
                tokens.extend(parse_list(v, new_prefix))
            else:
                tokens.append(f"{new_prefix}:{v}")
        return tokens

    def parse_list(lst, prefix):
        tokens = []
        for item in lst:
            if isinstance(item, dict):
                line_tokens = []
                for k, v in item.items():
                    if prefix_list_of_dict:
                        line_tokens.append(f"{prefix}-{k}:{v}")
                    else:
                        line_tokens.append(f"{k}:{v}")
                tokens.append(", ".join(line_tokens))
            else:
                tokens.append(f"{prefix}:{item}")
        return tokens

    tokens = parse_dict(data)
    return "\n".join(tokens)


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


def token2jsonV2(token_str,  prefix_list_of_dict=False):
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


def mapping_dict_keys(obj: Any, key_mapping: Dict):
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


def demo_json2token_v2(json_data, prefix_list_of_dict=False):
    # 示例使用
    token_str = json2tokenV2(json_data, sort_json_key=False, prefix_list_of_dict=prefix_list_of_dict)
    print(f"json_to_tokens:{token_str}")
    result = token2jsonV2(token_str, prefix_list_of_dict=prefix_list_of_dict)
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
        ],
        "嵌套0": [
            {"项目": "检查费", "金额": "50.00"},
            {"项目": "化验费", "金额": "60.00"}
        ],
        "嵌套1": {
            "嵌套2": [
                {"项目": "检查费", "金额": "70.00"},
                {"项目": "化验费", "金额": "80.00"}
            ]
        }
    }
    demo_json2token_v1(json_data)
    print("**" * 20)
    demo_json2token_v2(json_data, prefix_list_of_dict=True)
    print("**"*20)
    demo_json2token_v2(json_data, prefix_list_of_dict=False)