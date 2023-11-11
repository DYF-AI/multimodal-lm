import os.path
import time
import json
import multiprocessing
from tqdm import tqdm
from multimodallm.utils.file_utils import getAllFiles

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

from multimodallm.utils.image_utils import load_image
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=True,
    ocr_version="PP-OCRv4",
    det_db_box_thresh=0.1,
    det_db_thresh=0.1,

)

def ocr_predict(image_file:str):
    image_data, _ = load_image(image_file, return_chw=False)
    text_list = ocr.ocr(image_data, cls=True)
    if text_list[0] is None: return []
    ocr_result = []
    for text in text_list[0]:
        ocr_result.append(
            {
                "transcription": text[1][0],
                "points": text[0],
                "score":  text[1][1],
                "difficult": "false"
            }
        )
    return (image_file, ocr_result)

def get_predicted_file(file:str):
    predicted_files = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.strip().split("\t")
            predicted_files.append(split_line[0])
    return predicted_files


def run_ocr_predict(image_root:str, use_mp=True):
    all_image_files = getAllFiles(image_root, [".jpg", ".jpeg", ".gif", ".png", ".bmp"])
    predicted_files = get_predicted_file(r"J:\data\mllm-data\image-book\Label_continue_v3.txt")
    to_predict_image_files = []
    for imge_file in tqdm(all_image_files):
        image_file_name_split = imge_file.split("\\")
        image_file_name = image_file_name_split[-2] + "/" + image_file_name_split[-1]
        if image_file_name in predicted_files:
            print(f"{image_file_name} has predicted, continue!")
            continue
        to_predict_image_files.append(imge_file)
    all_image_files = to_predict_image_files

    all_ocr_reulsts = []
    if not use_mp:
        save_label = os.path.join(image_root, "Label.txt")
        save_fileState = os.path.join(image_root, "fileState.txt")
        # 数据量较大,边度边写,避免中途崩掉,没记录
        writer1 = open(save_label, "w", encoding="utf-8")
        writer2 = open(save_fileState, "w", encoding="utf-8")
        for index, image_file in enumerate(tqdm(all_image_files)):
            print(image_file)
            ocr_result = ocr_predict(image_file)
            if len(ocr_result) == 0: continue
            image_file_name_split = ocr_result[0].split("\\")
            image_file_name = image_file_name_split[-2] + "/" + image_file_name_split[-1]
            writer1.write(image_file_name + "\t" + json.dumps(ocr_result[1], ensure_ascii=False) + "\n")
            writer2.write(ocr_result[0] + "\t" + str(1) + "\n")
            #all_ocr_reulsts.append((image_file, ocr_result))
            if index%1000 == 0 or index == len(all_image_files)-1:
                writer1.flush()
                writer2.flush()

    else:
        pool = multiprocessing.Pool(processes=6)
        result_queue = multiprocessing.Manager().Queue()
        for image_file in tqdm(all_image_files):
            pool.apply_async(func=ocr_predict, args=(image_file,), callback=result_queue.put)
            time.sleep(0.5)
        pool.close()
        pool.join()

        num_done = 0
        for file in all_image_files:
            ocr_result = result_queue.get()
            print(f'file:{file}, Result:{ocr_result}')
            num_done += 1
            all_ocr_reulsts.append((file, ocr_result))
    return all_ocr_reulsts

def test_ocr_predict():
    image_file = r"J:\data\mllm-data\crawler-data\image\电脑配置单\0df431adcbef76091cf9b1eb2edda3cc7dd99ed7.jpg"
    result = ocr_predict(image_file=image_file)
    print(result)


def test_run_ocr_predict():
    image_root = r"J:\data\mllm-data\image-book"
    #image_root = r"J:\data\mllm-data\crawler-data\image\法律法规"
    all_ocr_reulsts = run_ocr_predict(image_root, use_mp=False)

    save_label = os.path.join(image_root, "Label1.txt")
    save_fileState = os.path.join(image_root, "fileState1.txt")

    writer1 = open(save_label, "w", encoding="utf-8")
    writer2 = open(save_fileState, "w", encoding="utf-8")
    # ppocrlabel-pil format
    for result in tqdm(all_ocr_reulsts):
        if len(result[0]) == 0: continue
        label_file_name_split = result[0].split("\\")
        label_file_name = label_file_name_split[-2] + "/" + label_file_name_split[-1]
        writer1.write(label_file_name + "\t" + json.dumps(result[1][1], ensure_ascii=False) + "\n")
        writer2.write(result[0] + "\t" + str(1) + "\n")
        writer1.flush()
        writer2.flush()
    writer1.close()
    writer2.close()


if __name__ == "__main__":
    # test_ocr_predict()
    test_run_ocr_predict()










