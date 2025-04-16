

import time
from lmdeploy.vl import load_image
from lmdeploy import pipeline
from lmdeploy.serve.openai.api_client import APIClient


model_path = "/mnt/n/model/Qwen/Qwen2-VL-2B-Instruct"


pipe = pipeline(model_path)
print(pipe)

while True:
    #image_path = input("请输入图片路径:")
    #prompt = input("请输入prompt:")
    input("输入")
    image_path = "/mnt/n/data/coco_2014_caption/validation/3480.jpg"
    prompt = "请详细描述一下这张图片"   # 推理速度大概也是67token/s, 和V100相差不大
    t1 = time.time()
    image = load_image(image_path)
    print(f"load_image_time: {time.time() - t1}")
    response = pipe(("请详细描述一下这张图片", image))
    print(response, response)
    t2 = time.time() - t1
    print(f"inference_time: {t2}, {response.generate_token_len/t2}")