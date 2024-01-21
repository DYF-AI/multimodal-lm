import requests
import wget
import os


url = "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-50_zh_middle_0011.jsonl"
save_dir = "M:/data/SkyPile-150B/data"

import requests


# def download_file(url, save_path):
#     response = requests.get(url, stream=True)
#     response.raise_for_status()  # 如果请求失败，抛出异常
#
#     with open(save_path, 'wb') as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             if chunk:
#                 f.write(chunk)
#
#             # 使用示例
#
#
# #url = 'http://example.com/file.txt'  # 替换为实际的URL
# save_path = os.path.join(save_dir, os.path.basename(url)) # 替换为实际的保存路径
# download_file(url, save_path)

# url_root = "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/"
# with open("skypile-150B-url-huggingface.txt", "r", encoding="utf-8") as f1, \
#     open("skypile-150B-url.txt", "w", encoding="utf-8") as f2:
#     for line in f1:
#         line = line.strip()
#         if line.endswith(".jsonl"):
#             f2.write(f"{url_root}{line}")
#             f2.write("\n")

