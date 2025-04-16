# Qwen2-vLLM-Local.py
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 设置环境变量
#os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# 模型ID：我们下载的模型权重文件目录
model_dir = "/mnt/n/model/Qwen/Qwen2-VL-2B-Instruct"

# Tokenizer初始化
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)

# 初始化大语言模型
llm = LLM(
    model=model_dir,
    tensor_parallel_size=1,  # CPU无需张量并行
    device='cuda',
)

# 超参数：最多512个Token
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Prompt提示词
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {
        'role': 'user',
        'content': [
            {
                'type':'image_url',
                'image_url':{
                    'url': '/mnt/n/data/coco_2014_caption/validation/3480.jpg'
                }
            },
            {'type':'text', 'text':'请详细描述一下这张图片'}
        ]
    }
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 模型推理输出
outputs = llm.generate([text], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    print(f'Prompt提示词: {prompt!r}, 大模型推理输出: {generated_text!r}')