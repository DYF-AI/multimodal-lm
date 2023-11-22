
import re
import numpy as np
from typing import Any

from multimodallm.utils.data_utils import json2token, token2json

# def json2token(obj:Any, sort_key: bool=True):
#     if isinstance(obj, list):
#         return r"<sep/>".join([json2token(v, sort_key) for v in obj])
#     elif isinstance(obj, dict):
#         items = sorted(obj.items(), key=lambda x:x[0]) if sort_key else obj.items()
#         return "".join([fr"<s_{k}>" + json2token(v, sort_key) + fr"</s_{k}>" for k,v in items])
#     obj = str(obj)
#     return obj

#
# new_special_tokens = [] # new tokens which will be added to the tokenizer
#
# def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
#     """
#     Convert an ordered JSON object into a token sequence
#     """
#     if type(obj) == dict:
#         if len(obj) == 1 and "text_sequence" in obj:
#             return obj["text_sequence"]
#         else:
#             output = ""
#             if sort_json_key:
#                 keys = sorted(obj.keys(), reverse=True)
#             else:
#                 keys = obj.keys()
#             for k in keys:
#                 if update_special_tokens_for_json_key:
#                     new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
#                     new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
#                 output += (
#                     fr"<s_{k}>"
#                     + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
#                     + fr"</s_{k}>"
#                 )
#             return output
#     elif type(obj) == list:
#         return r"<sep/>".join(
#             [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
#         )
#     else:
#         # excluded special tokens for now
#         obj = str(obj)
#         if f"<{obj}/>" in new_special_tokens:
#             obj = f"<{obj}/>"  # for categorical special tokens
#         return obj
#
# def convert_json_key_to_id(obj, key_ids):
#     if isinstance(obj, list):
#         return [convert_json_key_to_id(v, key_ids) for v in obj]
#     elif isinstance(obj, dict):
#         new_obj = dict()
#         for k, v in obj.items():
#             if k not in key_ids:
#                 key_ids[k] = max(key_ids.values())+1
#             new_obj[key_ids[k]] = convert_json_key_to_id(v, key_ids)
#         return obj
#     return obj
#
#
# def token2json(tokens, is_inner_value=False, expand_vocab=None):
#     """
#     Convert a (generated) token sequence into an ordered JSON format.
#     expand_vocab: 是扩展的特殊字符
#     """
#     output = {}
#
#     while tokens:
#         start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
#         if start_token is None:
#             break
#         key = start_token.group(1)
#         end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
#         start_token = start_token.group()
#         if end_token is None:
#             tokens = tokens.replace(start_token, "")
#         else:
#             end_token = end_token.group()
#             start_token_escaped = re.escape(start_token)
#             end_token_escaped = re.escape(end_token)
#             # 换行/n 需要增加re.S
#             content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE|re.S)
#             if content is not None:
#                 content = content.group(1).strip()
#                 if r"<s_" in content and r"</s_" in content:  # non-leaf node
#                     value = token2json(content, is_inner_value=True, expand_vocab=expand_vocab)
#                     if value:
#                         if len(value) == 1:
#                             value = value[0]
#                         output[key] = value
#                 else:  # leaf nodes
#                     output[key] = []
#                     for leaf in content.split(r"<sep/>"):
#                         leaf = leaf.strip()
#                         if leaf in expand_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
#                             leaf = leaf[1:-2]  # for categorical special tokens
#                         output[key].append(leaf)
#                     if len(output[key]) == 1:
#                         output[key] = output[key][0]
#
#             tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
#             if tokens[:6] == r"<sep/>":  # non-leaf nodes
#                 return [output] + token2json(tokens[6:], is_inner_value=True, expand_vocab=expand_vocab)
#
#     if len(output):
#         return [output] if is_inner_value else output
#     else:
#         return [] if is_inner_value else {"text_sequence": tokens}


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
