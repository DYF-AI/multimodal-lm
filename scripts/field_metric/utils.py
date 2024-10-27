from collections import defaultdict
from typing import Any, Dict, List
import difflib

def compute_one_metric(pred_dict: Dict[str, Any], gt_dict: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    # Initialize match info dictionary
    match_info = defaultdict(lambda: {"right_num": 0, "pred_num": 0, "gt_num": 0})

    # Flatten dictionaries and compute matches
    def flatten_dict(data: Dict[str, Any], parent_key: str = '') -> List[Dict[str, Any]]:
        """Flatten a nested dictionary with list of dicts"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}-{k}" if parent_key else k
            if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                for idx, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{new_key}-{idx}"))
            else:
                items.append((new_key, v))
        return items

    # Find similarity between two lists of dicts
    def find_best_match(pred_list: List[Dict[str, str]], gt_list: List[Dict[str, str]], base_key: str):
        gt_used = set()
        for p_item in pred_list:
            best_match = None
            best_score = 0
            for idx, g_item in enumerate(gt_list):
                if idx in gt_used:
                    continue
                # Calculate the similarity score based on matched key-value pairs
                #score = sum(1 for k, v in p_item.items() if g_item.get(k) == v)
                p_item_group = "-".join([v for k, v in p_item.items()])
                g_item_group = "-".join([g_item[k] for k in p_item.keys()])
                score = difflib.SequenceMatcher(None, p_item_group, g_item_group).quick_ratio()

                if score > best_score:
                    best_score = score
                    best_match = g_item
            if best_match:
                gt_used.add(gt_list.index(best_match))
                # Update match info for matching keys
                for key in p_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info[full_key]["pred_num"] += 1
                    match_info[full_key]["gt_num"] += 1
                    if p_item[key] == best_match.get(key):
                        match_info[full_key]["right_num"] += 1
            else:
                for key in p_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info[full_key]["pred_num"] += 1

        # Update match info for any remaining gt items not matched
        for idx, g_item in enumerate(gt_list):
            if idx not in gt_used:
                for key in g_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info[full_key]["gt_num"] += 1

    # Perform field-wise comparison
    for key in set(pred_dict.keys()).union(set(gt_dict.keys())):
        pred_val = pred_dict.get(key)
        gt_val = gt_dict.get(key)

        if isinstance(pred_val, list) and all(isinstance(i, dict) for i in pred_val) and \
                isinstance(gt_val, list) and all(isinstance(i, dict) for i in gt_val):
            # Flatten and compare lists of dictionaries
            find_best_match(pred_val, gt_val, key)
        else:
            match_info[key]["pred_num"] = int(pred_val is not None)
            match_info[key]["gt_num"] = int(gt_val is not None)
            if pred_val == gt_val:
                match_info[key]["right_num"] = 1 if pred_val is not None else 0

    return match_info


def demo1():
    # Input data
    pred = {
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
        ]
    }

    gt = {
        "材料类型": "电子发票",
        "票据代码": "50068122",
        "票据号码": "8250626222",
        "校验码": "edfdf4",
        "交款人": "冯昱硕",
        "开票日期": "2023年03月16日",
        "费用明细": [
            {"细项名称": "卫生材料费", "金额": "8.98"},
            {"细项名称": "治疗费", "金额": "3.4"},
        ],
        "嵌套0": [
            {"项目": "检查费", "金额": "50.00"},
            {"项目": "化验费", "金额": "60.00"}
        ]
    }

    # Run comparison
    match_info = compute_one_metric(pred, gt)

    # Print results
    #print(json.dumps(match_info, ensure_ascii=False, indent=4))
    print(match_info)


if __name__ == '__main__':
    demo1()