import os
import torch
import requests
import numpy as np
from PIL import Image
import gradio as gr
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig


if __name__ == "__main__":
    # config
    max_length = 2560
    image_size = [1024, 1024]
    MP = "J:/model/mllm-model/donut-pretrain/20240102/pl-checkpoint-232000-ned-0.8460975410122905"
    #MP = "G:/dongyongfei786/multimodal-lm/notebooks/tokenizer-zh/donut-zh-model"

    # processor
    processor = DonutProcessor.from_pretrained(MP)
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False

    # model
    model = VisionEncoderDecoderModel.from_pretrained(MP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"model:{model}")

    # prompt
    # token_prompt = "<s_cord-v2>"
    token_prompt = "<s_ocr_pretrain>"
    decoder_input_ids = processor.tokenizer(token_prompt,
                                            add_special_tokens=False,
                                            return_tensors="pt").input_ids.to(device)
    print(f"decoder_input_ids:{decoder_input_ids}")

    model.eval()
    def predict(input_image):
        pixel_values = processor.image_processor(input_image, return_tensors="pt").pixel_values.to(device)
        outputs = model.generate(pixel_values=pixel_values,
                                 decoder_input_ids=decoder_input_ids,
                                 max_length=model.decoder.config.max_position_embeddings,
                                 early_stopping=True,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 eos_token_id=processor.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True
                                 )
        prediction = processor.batch_decode(outputs.sequences)[0]
        #return str(processor.token2json(prediction))
        return processor.token2json(prediction)


    # test
    #image_path = "J:/dataset/mllm-data/mllm-finetune-data/trainticket/test/IMG_3840.jpg"
    image_path = "2_Book2_5_in_close_10.jpg"
    image_data = Image.open(image_path).convert("RGB")

    output = predict(image_data)
    print(output)

    run_gradio = True
    if run_gradio:
        demo = gr.Interface(fn=predict,
                            inputs=gr.inputs.Image(type="pil"),
                            outputs="json",
                            examples=[[image_path]]
                            )
        demo.launch(share=True)