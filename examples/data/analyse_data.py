
import pandas as pd

metadata_file = r"J:\data\mllm-data\mllm-pretrain-data\mllm-data-20231116.csv"

df = pd.read_csv(metadata_file)

length_list = []
for index, row in df.iterrows():
    text = row["ocr成行"]
    length_list.append(len(text))

length_num_dict = dict()
for length in length_list:
    if length not in length_num_dict:
        length_num_dict[length] = 0
    length_num_dict[length] += 1

print(length_num_dict)
