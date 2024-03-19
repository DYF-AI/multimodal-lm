from tqdm import tqdm
from transformers import AutoTokenizer
from vocabulary_pruner import VocabularyPruner, DonutVocabularyPruner, BloomVocabularyPruner


MP1 = "J:/model/bigscience-bloom-560m"
MP2 = "J:/model/YeungNLP-bloom-396m-zh"

# save_path = 'bloom-396m-zh'
# pruner = BloomVocabularyPruner()
# # 裁剪
# pruner.prune(MP1, MP2, save_path)
# # 检查裁剪的模型与原模型是否一致
# pruner.check(MP1, save_path, text='长风破浪会有时')

import langid
MP3 = "J:/model/mllm-model/donut-pretrain/20240102/pl-checkpoint-232000-ned-0.8460975410122905"

tokenizer3 = AutoTokenizer.from_pretrained(MP3)

## 统计词表中的语言分布
lang_dist, lang_dist_list = {}, {}
for idx, voc in enumerate(tqdm(tokenizer3.get_vocab())):
    r = langid.classify(voc)[0]
    if r not in lang_dist:
        lang_dist_list[r] = [0, []]
        lang_dist[r] = 0
    lang_dist_list[r][0] += 1
    lang_dist[r] += 1
    lang_dist_list[r][1].append(voc)
    if idx <= 10:
        print(voc, r)

import json
lang_dist = dict(sorted(lang_dist.items(), key=lambda x:x[1], reverse=True))
lang_dist_list = dict(sorted(lang_dist_list.items(), key=lambda x:x[1], reverse=True))

lang_dist.keys(), lang_dist

with open("lang_dist.json", "w", encoding="utf-8") as f1, \
    open("lang_dist_list.json", "w", encoding="utf-8") as f2:
    f1.write(json.dumps(lang_dist, ensure_ascii=False, indent=2))
    f2.write(json.dumps(lang_dist_list, ensure_ascii=False, indent=2))

tokenizer3_file = "J:/model/mllm-model/donut-pretrain/20240102/pl-checkpoint-232000-ned-0.8460975410122905/tokenizer.json"
tokenizer3_config_file = "J:/model/mllm-model/donut-pretrain/20240102/pl-checkpoint-232000-ned-0.8460975410122905/tokenizer_config.json"

with open(tokenizer3_file, "r", encoding="utf-8") as f1,\
        open(tokenizer3_config_file, "r", encoding="utf-8") as f2:
    tokenizer_data = json.load(f1)
    tokenizer3_config_data = json.load(f2)

vocab = tokenizer_data["model"]["vocab"]
len(vocab), vocab[100]

import re
def is_chinese_or_english(token):
    pattern = re.compile("^[\u4e00-\u9fa5a-zA-Z]+$")
    return bool(pattern.match(token))

def is_valid_token(token):
    """
    中文、英文、数字、符号
    :param token:
    :return:
    """
    #pattern = re.compile("^[\u4e00-\u9fa5a-zA-Z0-9\s`~!@#$%^&Ω🔞*()-_+={}[\]:;\"'<>,.?/]+$")
    pattern = re.compile("^[\u4e00-\u9fa5a-zA-Z0-9\s`~!@#$%^&Ω🔞*()≈≡≠＝<>＜＞≮≯∷±＋－×÷／∫∮∝∞∧∨∑∏∪∩∈∵∴⊥‖∠⌒≌∽√（）【】｛｝ⅠⅡ⊕⊙∥αβγδεζηθΔ+={}[\]:;\"'<>,.?/]+$")

    return bool(pattern.match(token))

input_str = "3.2"
is_chinese_or_english(input_str), is_valid_token(input_str), langid.classify(input_str)


new_vocab, filtered_vocab = [], []
for v in tqdm(vocab):
    token = v[0].replace("▁", "")
    lang = langid.classify(token)
    is_ch_or_en = is_chinese_or_english(token)
    is_valid = is_chinese_or_english(token)
    flag = True if lang[0] in ["en", "zh"] else False
    if is_ch_or_en or is_valid or flag:
        new_vocab.append(v)
    else:
        filtered_vocab.append(v)

len(new_vocab), len(filtered_vocab)
with open("new_vocab.txt", "w", encoding="utf-8") as f1,\
    open("filtered_vocab.txt", "w", encoding="utf-8") as f2:
    vocab_1 = [v[0] for v in new_vocab]
    vocab_2 = [v[0] for v in filtered_vocab]
    f1.write("\n".join(vocab_1))
    f2.write("\n".join(vocab_2))

added_tokens = tokenizer_data["added_tokens"]
new_add_tokens = []
new_added_tokens_start_id = len(new_vocab) - 1
for t in added_tokens:
    if t["id"] in [0, 1, 2, 3]:
        new_add_tokens.append(t)
        continue
    t["id"] = new_added_tokens_start_id
    new_add_tokens.append(t)
    new_added_tokens_start_id += 1

import copy
new_tokenizer_data = copy.deepcopy(tokenizer_data)
new_tokenizer_data["model"]["vocab"] = new_vocab
new_tokenizer_data["added_tokens"] = new_add_tokens

new_tokenizer_config_data = copy.deepcopy(tokenizer3_config_data)
new_tokenizer_config_data["added_tokens"] = new_add_tokens
new_tokenizer_config_data["added_tokens_decoder"] = new_add_tokens


with open("./donut-zh/tokenizer.json", "w", encoding="utf-8") as f1,\
        open("./donut-zh/tokenizer_config.json", "w", encoding="utf-8") as f2:
    f1.write(json.dumps(new_tokenizer_data, ensure_ascii=False, indent=2))
    f2.write(json.dumps(new_tokenizer_config_data, ensure_ascii=False, indent=2))


MP4 = "./donut-zh"
tokenizer4 = AutoTokenizer.from_pretrained(MP4)

text = "中国平安是世界500强"
tokenizer3.tokenize(text), tokenizer4.tokenize(text)

save_path = 'donut-zh-model'
# pruner = BloomVocabularyPruner()
pruner = DonutVocabularyPruner()
# 裁剪
pruner.prune(MP3, MP4, save_path)