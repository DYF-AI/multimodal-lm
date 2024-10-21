import argparse
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from mllm.utils.file_utils import getAllFiles


def main(args):
    print(f"args:{args}")
    image_files = getAllFiles(args.image_folder, suffix_list=[".png", ".jpg"])
    print(f"len(image_files): {len(image_files)}")
    ocr_type = args.ocr_type
    assert ocr_type in ["ocr", "format"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                                      device_map='cuda',
                                      use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    query = "请抽取图片中的所有文字:"
    with open(args.save_res_path, "w", encoding="utf-8") as fo:
        for idx, image_file in enumerate(tqdm(image_files)):
            res = model.chat(tokenizer, image_file, ocr_type=ocr_type)
            row_data = {
                "query": query,
                "response": res,
                "image_path": [image_file]
            }
            fo.write(json.dumps(row_data, ensure_ascii=False) + "\n")
            if idx % 50 == 0:
                print(idx, row_data)
            fo.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="got-ocr-infer")
    parser.add_argument("--model_path", type=str, help="model-ckpt")
    parser.add_argument("--ocr_type", type=str, help="ocr_type")
    parser.add_argument("--image_folder", type=str, help="image_folder")
    parser.add_argument("--save_res_path", type=str, help="save_path")
    args = parser.parse_args()
    main(args)
