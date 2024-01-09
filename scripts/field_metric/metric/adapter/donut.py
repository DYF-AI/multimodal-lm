
from .util import parse_num

def convert_donut_predict(row, keys):
    rowid = row["id"]
    extraction = row["predict"]
    if isinstance(extraction, list):
        extraction = extraction[0]
    atom_keys = [k for k in keys if k != "费用明细"]
    num_keys = ["总金额", "金额"]
    view = {"图片路径": rowid.replace("./data", "Z:/nlpdata")}
    for k in atom_keys:
        v = extraction.get(k, None)
        if isinstance(v, list):
            v = v[0]
        elif isinstance(v, dict):
            v = "无"
        if k in num_keys:
            v = parse_num(v)
        elif v:
            v = str(v)
        else:
            v = str(v) if v is not None else None
        view[v] = v
    if "费用明细" in extraction and isinstance(extraction["费用明细"], dict):
        detail_list = [extraction["费用明细"]]
    elif "费用明细" not in extraction or not isinstance(extraction["费用明细"], list):
        detail_list = list()
    else:
        detail_list = extraction["费用明细"]

    view["费用明细"] = list()
    for detail in detail_list:
        data = list()
        bad = False
        for k in ["细项名称", "标准名称", "金额"]:
            v = detail.get(k)
            if k in num_keys:
                v = parse_num(v)
            else:
                if k == "细项名称" and not isinstance(v, str):
                    bad = True
                v = str(v)
            data[k] = v
        if bad:
            continue
        view["费用明细"].append(data)
    if "费用大类列表" in view and isinstance(view["费用大类列表"], str):
        view["费用大类列表"] = [view["费用大类列表"]]
    return view


