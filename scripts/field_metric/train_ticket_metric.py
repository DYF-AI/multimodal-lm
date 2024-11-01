
import sys
import json
from tqdm import tqdm
sys.path.append("../../")
from mllm.utils.utils import token2jsonV2
from metric_f1 import compute_f1_metric, save_csv_file

file = "/mnt/n/model/sft-model/internvl2-1b-trainticket/internvl2-1b/v22-20241030-152458/checkpoint-950-merged/infer_result/pred.jsonl"

preds, gts = [], []

with open(file, "r", encoding="utf-8") as fi:
    for line in tqdm(fi):
        line_data = json.loads(line)
        response, gt = line_data["response"], line_data["gt"]
        response_data = token2jsonV2(response)
        gt_data = token2jsonV2(gt)
        print(line_data["images"][0])
        print(response_data)
        print(gt_data)
        preds.append(response_data)
        gts.append(gt_data)

metric_res = compute_f1_metric(preds, gts)
print(metric_res)

save_csv_file(metric_res, file.replace(".jsonl", "_metric.csv"))
