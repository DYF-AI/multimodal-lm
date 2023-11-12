"""

 File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\paddleocr\tools\infer\predict_rec.py", line 619, in __call__
    output = output_tensor.copy_to_cpu()
numpy.core._exceptions.MemoryError: Unable to allocate 22.1 MiB for an array with shape (6, 146, 6625) and data type float32
"""
from tqdm import tqdm

def merge_ocr_result(ocr_result_1:str, ocr_result_2, save_merge_result:str):
    with open(ocr_result_1, "r", encoding="utf-8") as f1, \
        open(ocr_result_2, "r", encoding="utf-8") as f2, \
        open(save_merge_result, "w", encoding="utf-8") as f3:
        for line in tqdm(f1, desc="copy_file_1"):
            f3.write(line)
        for line in tqdm(f2, desc="copy_file_2"):
            f3.write(line)
        f3.flush()

if __name__ == "__main__":
    ocr_result_1 = r"J:\data\mllm-data\image-book\Label_continue_v3.txt"
    ocr_result_2 = r"J:\data\mllm-data\image-book\Label.txt"
    save_merge_result = r"J:\data\mllm-data\image-book\Label_continue_v4.txt"
    merge_ocr_result(ocr_result_1, ocr_result_2, save_merge_result)