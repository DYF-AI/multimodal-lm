import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLM4Tokenizer

device = "cuda"

MODEL_PATH = "N:/model/THUDM/glm-4v-9b"
#MODEL_PATH = "N:/model/THUDM/glm-4v-9b-4-bits"
IMAGE_PATH =  "N:/dataset/增值税发票/image/b3.jpg"

#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = ChatGLM4Tokenizer.from_pretrained(MODEL_PATH)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto',
# ).eval()

model = ChatGLMForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='auto',
).eval()

while True:
    #query = '这是一张增值税发票,请详细描述这张图片的内容'
    query = input("prompt:")
    image = Image.open(IMAGE_PATH).convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode

    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))
