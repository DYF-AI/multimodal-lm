import os
from tqdm import tqdm
import multiprocessing
from mllm.utils.image_utils import load_image, pdf_to_images
from mllm.utils.file_utils import getAllFiles

"""
    批量将pdf转成图片
"""

def test_pdf_to_image():
    pdf_file = "K:/doc/book/Python设计模式-第2版-[印]Chetan Giridhar-韩波译2017.pdf"
    save_image_path = "K:/doc/book"
    pdf_to_images(pdf_file, save_image_path)

def worker(pdf_file:str, save_path):
    print("worker computing...")
    flag = pdf_to_images(pdf_file, save_path)
    return flag

def run_pdf_to_images(book_path, save_image_path, use_mp=True):
    pdf_files = getAllFiles(book_path, [".pdf"])
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    if not use_mp:
        for pdf_file in tqdm(pdf_files):
            print(pdf_file)
            flag = pdf_to_images(pdf_file, save_image_path)
    else:
        pool = multiprocessing.Pool(processes=16)
        result_queue = multiprocessing.Manager().Queue()
        for file in tqdm(pdf_files):
            pool.apply_async(func=worker, args=(file,save_image_path), callback=result_queue.put)
        pool.close()
        pool.join()

        num_done = 0
        for file in pdf_files:
            result = result_queue.get()
            print(f'file:{file}, Result:{result}')
            num_done += 1


if __name__ == "__main__":
    run_pdf_to_images("J:/data/books", "K:/data/mllm-data/image-book")