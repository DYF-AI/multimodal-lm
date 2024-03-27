
import pandas as pd


metadata_file = "J:/data/mllm-data/mllm-pretrain-data/mllm-data-20231116.csv"

df = pd.read_csv(metadata_file)

length_list = []
for index, row in df.iterrows():
    text = row["ocr成行"]
    try:
        length_list.append(len(text))
    except Exception as e:
        print(e)
        continue

length_num_dict = {
    "<1000":0,
    "1000<=x<2560":0,
    ">=2560":0
}
for length in length_list:
    if length not in length_num_dict:


        if length < 1000:
            length_num_dict["<1000"] +=1
        elif length < 2560:
            length_num_dict["1000<=x<2560"] +=1
        else:
            length_num_dict[">=2560"] += 1


print(length_num_dict)

#
# import seaborn as sb
# import matplotlib as plt
#
# df["ocr成行"] = list(map(lambda x: len(x) if type(x) == "str" else 0, df["ocr成行"]))
#
# sb.countplot("seq lenght", data=df)
#
# plt.xticks([])
# plt.show()