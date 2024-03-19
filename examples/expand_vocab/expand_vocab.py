import os
from tqdm import tqdm
import pandas as pd

"""
    
"""

def read_csv(csv_file):
    data_list = list()
    df = pd.read_csv(csv_file)
    for index, row in tqdm(df.iterrows()):
        text = row["ocr成行"]
        data_list.append(text)
        #if index > 10: break
    return data_list

def read_txt(txt_file):
    data_list = list()
    with open(txt_file, "r", encoding="utf-8") as f1:
        for index, line in enumerate(tqdm(f1)):
            data_list.append(line.strip())
            if index > 10: break
    return data_list

def find_unk_words(data_list:list, tokenizer):
    unk_words = dict()
    for text in tqdm(data_list):
        # tokenizer(data_list) 会报错,带排查 TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
        try:
            encoded = tokenizer(text, return_attention_mask=False, return_offsets_mapping=True)
            for input_id, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
                if input_id != tokenizer.unk_token_id:
                    continue
                word = text[start:end]
                if word not in unk_words:
                    unk_words[word] = 0
                unk_words[word] += 1
        except Exception as e:
            print(f"text:{text}, tokenize error:{e}, ")
            continue
    return unk_words

def find_unk_chars(unk_words, tokenizer):
    unk_chars = dict()
    vocab = tokenizer.vocab
    for word, freq in tqdm(unk_words.items()):
        for ch in set(word):
            if ch in vocab:
                continue
            if ch not in unk_chars:
                unk_chars[ch] = 0
            unk_chars[ch] += freq
    return unk_chars

def writer_expand_vocab(tokenizer):
    DP1 = "J:/data/mllm-data/mllm-pretrain-data/mllm-data-20231116.csv"  # csv
    DP2 = "J:/data/corpus-data/Chinese-Names-Corpus/Chinese_Dict_Corpus/ChengYu_Corpus（5W）.txt"
    DP3 = "J:/data/corpus-data/Chinese-Names-Corpus/Chinese_Names_Corpus/Ancient_Names_Corpus（25W）.txt"
    DP4 = r"J:/data/corpus-data/Chinese-Names-Corpus/Chinese_Names_Corpus/Chinese_Names_Corpus（120W）.txt"

    data_list = list()
    data_list_1 = read_csv(DP1)
    data_list.extend(data_list_1)
    for mp in [DP2, DP3, DP4]:
        data_list.extend(read_txt(mp))
    print(data_list[:10])

    unk_words = find_unk_words(data_list, tokenizer)
    unk_chars = find_unk_chars(unk_words, tokenizer)
    expand_vocab = list(unk_chars.keys())
    print(f"expand_vocab:{expand_vocab}, {len(expand_vocab)}")

    with open("expand_vocab.txt", "w", encoding="utf-8") as f1:
        for vocab in expand_vocab:
            f1.write(vocab + "\n")

if __name__ == "__main__":
    MP = "J:/model/pretrained-model/torch\donut-base"
    NEW_MP = "J:/model/pretrained-model/torch/donut-base-expand-vocab"
    if os.path.exists(NEW_MP):
        os.makedirs(NEW_MP)

    from transformers import DonutProcessor, VisionEncoderDecoderModel,VisionEncoderDecoderConfig
    processor = DonutProcessor.from_pretrained(MP)
    config = VisionEncoderDecoderConfig.from_pretrained(MP)

    model = VisionEncoderDecoderModel.from_pretrained(MP, config=config)
    tokenizer = processor.tokenizer

    expand_vocab_exists = True
    if not expand_vocab_exists:
        writer_expand_vocab(tokenizer)

    assert os.path.exists("expand_vocab.txt")

    with open("expand_vocab.txt", "r", encoding="utf-8") as f1:
        expand_vocab_list = [line.strip() for line in f1]
    print(f"add vocab num:{len(expand_vocab_list)}")
    newly_added_num = processor.tokenizer.add_tokens(expand_vocab_list)
    if newly_added_num > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.save_pretrained(NEW_MP)
    processor.save_pretrained(NEW_MP)
    print("expand vocabm, Done!")




