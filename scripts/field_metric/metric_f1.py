from collections import defaultdict, OrderedDict
from typing import Any, Dict, List
import difflib
import pandas as pd

def compute_one_metric(pred_dict: Dict[str, Any], gt_dict: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    # Initialize match info dictionary
    match_info_one = defaultdict(lambda: {"right_num": 0, "pred_num": 0, "gt_num": 0})

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
                p_item_group = "-".join([v if v is not None else "" for k, v in p_item.items()])
                g_item_group = "-".join([g_item[k] if g_item[k] is not None else ""  for k in p_item.keys()])
                score = difflib.SequenceMatcher(None, p_item_group, g_item_group).quick_ratio()

                if score >= best_score:
                    best_score = score
                    best_match = g_item
            if best_match:
                gt_used.add(gt_list.index(best_match))
                # Update match info for matching keys
                for key in p_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info_one[full_key]["pred_num"] += 1
                    match_info_one[full_key]["gt_num"] += 1
                    if p_item[key] == best_match.get(key):
                        match_info_one[full_key]["right_num"] += 1
            else:
                for key in p_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info_one[full_key]["pred_num"] += 1

        # Update match info for any remaining gt items not matched
        for idx, g_item in enumerate(gt_list):
            if idx not in gt_used:
                for key in g_item.keys():
                    full_key = f"{base_key}-{key}"
                    match_info_one[full_key]["gt_num"] += 1

    # Perform field-wise comparison
    #for key in set(pred_dict.keys()).union(set(gt_dict.keys())):
    all_keys = list(OrderedDict.fromkeys(list(pred_dict.keys()) + list(gt_dict.keys())))
    for key in all_keys:
        gt_val = gt_dict.get(key)
        pred_val = pred_dict.get(key)
        if isinstance(gt_val, list) and not pred_val:
            pred_val = [{k: None for k in item} for item in gt_val]
        if isinstance(pred_val, list) and not gt_val:
            gt_val = [{k: None for k in item} for item in pred_val]

        if isinstance(pred_val, list) and all(isinstance(i, dict) for i in pred_val) and \
                isinstance(gt_val, list) and all(isinstance(i, dict) for i in gt_val):
            # Flatten and compare lists of dictionaries
            find_best_match(pred_val, gt_val, key)
        else:
            match_info_one[key]["pred_num"] = int(pred_val is not None)
            match_info_one[key]["gt_num"] = int(gt_val is not None)
            if pred_val == gt_val:
                match_info_one[key]["right_num"] = 1 if pred_val is not None else 0

    return match_info_one


def compute_f1_metric(preds:list, gts:list):
    assert len(preds) == len(gts)
    total_right_num, total_pred_num, total_gt_num = 0, 0, 0
    match_info = defaultdict(lambda: {"right_num": 0, "pred_num": 0, "gt_num": 0, "precision": 0., "recall": 0., "f1": 0.})
    # calculate
    for pred, gt in zip(preds, gts):
        match_info_one = compute_one_metric(pred, gt)
        for k1, v1 in match_info_one.items():
            total_right_num += v1["right_num"]
            total_pred_num += v1["pred_num"]
            total_gt_num += v1["gt_num"]
            for k2, v2 in v1.items():
                match_info[k1][k2] += v2
            match_info[k1]["precision"] = round(match_info[k1]["right_num"] / (match_info[k1]["pred_num"] + 1e-6), 4)
            match_info[k1]["recall"] = round(match_info[k1]["right_num"] / (match_info[k1]["gt_num"] + 1e-6), 4)
            match_info[k1]["f1"] = round(2*match_info[k1]["precision"]*match_info[k1]["recall"]/(match_info[k1]["precision"] + match_info[k1]["recall"] + 1e-6), 4)
    # calculate precision, recall, f1
    match_info["total"]["right_num"] = total_right_num
    match_info["total"]["pred_num"] = total_pred_num
    match_info["total"]["gt_num"] = total_gt_num
    p, r = total_right_num / total_pred_num, total_right_num / total_gt_num
    match_info["total"]["precision"] = p
    match_info["total"]["recall"] = r
    match_info["total"]["f1"] = 2*p*r/(p+r)
    return match_info


def save_csv_file(match_info, save_file):
    match_data = {
        "field": [], "right_num": [], "pred_num": [], "gt_num": [],
        "precision": [], "recall": [], "f1": []
    }
    for k1, v1 in match_info.items():
        match_data["field"].append(k1)
        for k2, v2 in v1.items():
            match_data[k2].append(v2)
    match_data_df = pd.DataFrame(match_data)
    match_data_df.to_csv(save_file, index=False)
    #match_data_df.to_excel(save_file, index=False)

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
    # match_info_one = compute_one_metric(pred, gt)
    match_info = compute_f1_metric([{}, pred, pred], [gt, gt, gt])

    save_csv_file(match_info, "test.csv")

    # Print results
    #print(json.dumps(match_info, ensure_ascii=False, indent=4))
    print(match_info)


if __name__ == '__main__':
    demo1()