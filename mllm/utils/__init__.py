
import re
import numpy as np
from typing import Any
import sys
sys.path.append("../../")
from mllm.utils.data_utils import json2token, token2json
from mllm.utils.utils import *


def cal_model_parameters(model):
    model_params = sum(p.numel() for p in model.parameters())
    return model_params


if __name__ == "__main__":
    #sequence = '{"gt_parse": {"text_sequence": "折页传单画册\\nCD\\nCD\\nDigi Pack\\nDigiPack\\n低至\\n工厂直销 买贵包退\\n1\\n专业印刷 免费设计\\n夫 /份"}}'
    sequence = '{"gt_parse": {"text_sequence": "折页传单画册</n>CD</n>CD</n>Digi Pack</n>DigiPack</n>低至</n>工厂直销 买贵包退</n>1</n>专业印刷 免费设计</n>夫 /份"}}'
    import json
    expand_vocab = ["<s_gt_parse>", "</s_gt_parse>", "<s_text_sequence>", "</s_text_sequence>"]
    json_seq = json.loads(sequence)
    seq = json2token(json_seq)
    json_seq1 = token2json(seq, expand_vocab=expand_vocab)
    MP = r"J:\model\pretrained-model\torch\donut-base"

    from transformers import DonutProcessor
    processor = DonutProcessor.from_pretrained(MP)
    processor.tokenizer.add_tokens(["</n>"])
    print(processor.tokenizer(seq))
    print(processor.tokenizer.convert_ids_to_tokens(processor.tokenizer(seq)["input_ids"]))
    print(json_seq1)
