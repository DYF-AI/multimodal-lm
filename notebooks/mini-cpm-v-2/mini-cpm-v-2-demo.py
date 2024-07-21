import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from modeling_minicpmv import MiniCPMV


MP = 'N:/model/openbmb/MiniCPM-V-2'

#model = AutoModel.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = MiniCPMV.from_pretrained(MP, torch_dtype=torch.bfloat16)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
model = model.to(device='cuda', dtype=torch.bfloat16)
# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
#model = model.to(device='cuda', dtype=torch.float16)
# For Mac with MPS (Apple silicon or AMD GPUs).
# Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
#model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
model.eval()


while True:
    image_file = "N:/dataset/增值税发票/image/b3.jpg"

    image = Image.open(image_file).convert('RGB')
    #question = 'What is in the image?'
    # question = '请问购买方是？'
    #question = '请详细描述这张图片'
    # query = '这是一张增值税发票,请详细描述这张图片的内容'
    question = input("prompt:")
    msgs = [{'role': 'user', 'content': question}]

    res, context, _ = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    print(res)