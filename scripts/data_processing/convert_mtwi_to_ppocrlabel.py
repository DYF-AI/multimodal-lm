import os
import json
from tqdm import tqdm
from multimodallm.utils.file_utils import getAllFiles


"""
    将J:\data\mllm-data\mtwi-2018-data\image数据转为，统一的数据格式Label.txt
    法律法规/zip_1618890966hWj1yE.jpg	[{"transcription": "药", "points": [[68, 73], [313, 73], [313, 325], [68, 325]], "difficult": false}]
    法律法规/7aaebb6d0adf44ed8fc04f43da96c41a.jpeg	[{"transcription": "中华人民共和国", "points": [[73, 457], [754, 457], [754, 553], [73, 553]], "difficult": false}, {"transcription": "和境保护法", "points": [[150, 602], [679, 602], [679, 704], [150, 704]], "difficult": false}]
"""

def convert(image_root:str, txt_root:str, save_label:str, save_state_file:str):
    image_files = getAllFiles(image_root, [".jpg", ".png"])
    txt_files = getAllFiles(txt_root, [".txt"])
    with open(save_label, "w", encoding="utf-8") as f1, open(save_state_file, "w", encoding="utf-8") as f2:
        for image_file in tqdm(image_files):
            print(image_file)
            image_file_splittext = os.path.splitext(image_file)
            print(image_file_splittext)
            ocr_txt_file = image_file_splittext[0].replace("mtwi-2018-image", "txt_train")+".txt"
            with open(ocr_txt_file, "r", encoding="utf-8") as f3:
                ocr_result = []
                for line in f3:
                    line_split = line.strip().split(",")
                    print(line_split)
                    file_name_split = image_file.split("/")
                    file_id = file_name_split[-2] + "/" + file_name_split[-1]
                    points = [[float(line_split[i]), float(line_split[i+1])] for i in range(0,8,2)]
                    if line_split[8] == "###":
                        continue
                    ocr_result.append(
                        {
                            "transcription": line_split[8],
                            "points":points,
                            "score": 0.99,
                            "difficult": "false"
                        }
                    )
            f1.write(file_id + "\t" + json.dumps(ocr_result, ensure_ascii=False) + "\n")
            f2.write(image_file + "\t" + str(1) + "\n")
            f1.flush()
            f2.flush()

if __name__ == "__main__":
    image_root = r"J:\data\mllm-data\mtwi-2018-data\mtwi-2018-image"
    txt_root = r"J:\data\mllm-data\mtwi-2018-data\txt_train"
    save_label = r"J:\data\mllm-data\mtwi-2018-data\mtwi-2018-image\Label.txt"
    save_state_file = r"J:\data\mllm-data\mtwi-2018-data\mtwi-2018-image\fileState.txt"
    convert(image_root, txt_root, save_label, save_state_file)
