import copy
import os.path
from typing import List
import time
import argparse
import torch
import torch.nn as nn
from PIL import Image
from transformers import LogitsProcessor, VisionEncoderDecoderModel, DonutProcessor, LogitsProcessorList, \
    VisionEncoderDecoderConfig
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

import transformers

print(f"transformers.__version__:{transformers.__version__}")


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.LongTensor) -> torch.LongTensor:
        @torch.jit.script_if_tracing
        def is_invalid(scores):
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

        scores = is_invalid(scores)
        return scores


def load_checkpoint_model(model_path: str,
                          image_size: list = None,
                          max_length: int = 768,
                          task_prompt="<s_ocr_pretrain>"):
    if image_size is None:
        image_size = [1024, 1024]
    config = VisionEncoderDecoderConfig.from_pretrained(model_path)
    config.encoder.image_size = image_size  # (height, width)
    config.decoder.max_length = max_length
    ignore_mismatched_sizes = False
    if max_length > config.decoder.max_position_embeddings:
        config.decoder.max_position_embeddings = max_length
        ignore_mismatched_sizes = True
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path,
                                                      config=config,
                                                      ignore_mismatched_sizes=ignore_mismatched_sizes)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([task_prompt])[0]
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False
    tokenzier = processor.tokenizer

    return config, model, model.encoder, model.decoder, processor, tokenzier


class ChatTokenModel(nn.Module):
    def __init__(self,
                 decoder_model,
                 max_length=2560,
                 temperature=0.95,
                 num_beams=1,
                 top_k=50,
                 top_p=0.7):
        super(ChatTokenModel, self).__init__()
        self.model = decoder_model
        self.max_length = max_length
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(InvalidScoreLogitsProcessor())

        # prepare logist_warper
        # self.logists_warper = LogitsProcessorList()
        # if temperature is not None and temperature != 1.0:
        #     self.logists_warper.append(TemperatureLogitsWarper(temperature))
        # min_tokens_to_keep = 2 if num_beams > 1 else 1
        # if top_k is not None and top_k != 0:
        #     self.logists_warper.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep))
        # if top_p is not None and top_p < 1.0:
        #     self.logists_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep))

    def forward(self,
                input_ids: torch.LongTensor,
                prepare_input_ids: torch.LongTensor,
                encoder_hidden_states: torch.LongTensor,
                attention_mask: torch.LongTensor,
                unfinished_sequence: torch.LongTensor,
                past_key_values: List[List[torch.LongTensor]] = None):
        pad_token_id = 1
        eos_token_id = [2]
        eos_token_id_tensor = torch.Tensor(eos_token_id).to(input_ids.device)

        score = None
        output_attentions = False
        output_hidden_states = False
        use_cache = True

        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
            "is_frist_forward": True,  # 随意, 后续不依赖该值
        }
        this_peer_finshed = torch.zeros([1])

        model_inputs = {
            "input_ids": prepare_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask
        }

        outputs = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            encoder_hidden_states=model_inputs["encoder_hidden_states"],
            past_key_values=model_inputs["past_key_values"],
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribute
        next_token_scores = self.logits_processor(input_ids, next_token_logits)
        # new_token_scores = self.logits_warper(input_ids, next_token_scores)

        next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        # if eos_token_id is not None:
        next_tokens = next_tokens * unfinished_sequence + pad_token_id * (1 - unfinished_sequence)

        # update generated ids, model_inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
        )

        @torch.jit.script_if_tracing
        def eos_stopping_criteria(unfinished_sequence: torch.Tensor):
            if unfinished_sequence.max().equal(torch.zeros([1], device=unfinished_sequence.device).max()):
                return torch.ones([1])
            return torch.zeros([1])

        @torch.jit.script_if_tracing
        def max_length_stopping_criteria(input_ids: torch.LongTensor, max_length: int):
            if input_ids.shape[-1] >= max_length:
                return torch.ones([1])
            return torch.zeros([1])

        @torch.jit.script_if_tracing
        def merge_stopping_criteria(this_peer_finished: torch.Tensor, stop_res: torch.Tensor):
            if torch.is_nonzero(stop_res):
                this_peer_finished = torch.ones([1])
            return this_peer_finished

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequence = unfinished_sequence.mul(
            next_tokens.tile(eos_token_id_tensor.size(0), 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        eos_stopping_res = eos_stopping_criteria(unfinished_sequence)
        max_length_stopping_res = max_length_stopping_criteria(input_ids, self.max_length)

        this_peer_finshed = merge_stopping_criteria(eos_stopping_res, max_length_stopping_res)

        return this_peer_finshed, input_ids, model_kwargs["attention_mask"], unfinished_sequence, model_kwargs[
            "past_key_values"]


class ChatLoopModel(nn.Module):
    def __init__(self, tokenModel, tokenModel2):
        super(ChatLoopModel, self).__init__()
        self.tokenModel = tokenModel
        self.tokenModel2 = tokenModel2

    def forward(self,
                input_ids: torch.LongTensor,
                encoder_hidden_states: torch.LongTensor,
                attention_mask: torch.LongTensor):
        prepare_input_ids = input_ids

        # keep track of which dequence are already finished
        unfinished_sequence = torch.ones(input_ids.size(0), dtype=torch.long, device=input_ids.device)

        # init param
        is_first_forward = torch.ones([1])
        past_key_values = torch.jit.annotate(List[List[torch.FloatTensor]], [])

        """
            loop
        """

        i = 0
        while True:
            if not torch.is_nonzero(is_first_forward):
                prepare_input_ids = input_ids[:, -1:]
            i += 1
            if torch.is_nonzero(is_first_forward):
                this_peer_finished, input_ids, attention_mask, unfinished_sequence, new_past_key_values = self.tokenModel(
                    input_ids=input_ids,
                    prepare_input_ids=prepare_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    unfinished_sequence=unfinished_sequence,
                )
            else:
                this_peer_finished, input_ids, attention_mask, unfinished_sequence, new_past_key_values = self.tokenModel2(
                    input_ids=input_ids,
                    prepare_input_ids=prepare_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    unfinished_sequence=unfinished_sequence,
                    past_key_values=past_key_values,
                )

            if torch.is_nonzero(this_peer_finished):
                break

            is_first_forward = torch.zeros([1])

            # g构造past_key_values
            tmp_past_key_values = torch.jit.annotate(List[List[torch.FloatTensor]], [])
            for value in new_past_key_values:
                tmp_past_key_values.append(list(value))
            past_key_values = tmp_past_key_values
        return input_ids


"""
    generateModel, 由它触发loop
"""


class ChatGenerateModel(nn.Module):
    def __init__(self, loopModel, origin_model, tokenizer):
        super(ChatGenerateModel, self).__init__()
        self.model = origin_model
        self.tokenizer = tokenizer
        self.loopModel = loopModel

    def forward(self,
                input_ids: torch.LongTensor,
                encoder_hidden_states: torch.LongTensor,
                attention_mask: torch.LongTensor,
                max_length: int,
                num_beams: int,
                do_sample: bool,
                top_p: float,
                temperature: float):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature
        }
        generation_config = self.model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        input_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, generation_config.bos_token_id, model_kwargs
        )

        model_kwargs["output_attnetions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["encoder_hidden_states"] = encoder_hidden_states

        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=False,
            **model_kwargs,
        )
        return self.loopModel(
            input_ids=input_ids,
            encoder_hidden_states=model_kwargs["encoder_hidden_states"],
            attention_mask=model_kwargs["attention_mask"]
        )


"""
    总模型：包装ChatGenerateModel
"""


class ChatModel(nn.Module):
    def __init__(self,
                 loopModel,
                 encoderModel,
                 origin_decoder_model,
                 tokenizer,
                 max_length=768,
                 num_beams=1,
                 do_sample=True,
                 top_p=0.7,
                 temperature=0.95):
        super(ChatModel, self).__init__()
        self.encoder_model = encoderModel
        self.model = origin_decoder_model
        self.tokenizer = tokenizer
        self.generateModel = ChatGenerateModel(loopModel, origin_decoder_model, tokenizer)
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature

    def forward(self, pixel_values, input_ids, attention_mask):
        encoder_hidden_states = self.encoder_model(pixel_values)["last_hidden_state"]
        inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask
        }
        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "temperature": self.temperature
        }
        outputs = self.generateModel(**inputs, **gen_kwargs)
        return outputs


def export_torchscript(parser=None):
    if parser is not None:
        args = parser.parse_args()
        MP = args.checkpoint_path
        SAVE_MP = args.save_torchscript_path
        max_length = args.max_length
        task_prompt = args.task_prompt
        image_size = args.image_size
        image_path = args.image_path
    else:
        MP = "/mnt/g/dongyongfei786/donut-tutorial/donut-save-hf/epoch_10_ned_0.008969717364156379"
        SAVE_MP = "./save_torchscript_path"
        max_length = 768
        task_prompt = "<s_cord-v2>"
        image_size = [1280, 960]
        image_path = "/mnt/n/data/ICDAR/ICDAR-2019/Task-SROIE/SROIE_test_images_task_3/X00016469670.jpg"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"device:{device}")

    with torch.no_grad():
        torch.manual_seed(123)
        if not os.path.exists(SAVE_MP):
            os.makedirs(SAVE_MP)
        config, model, encoder, decoder, processor, tokenzier = \
            load_checkpoint_model(MP, image_size=image_size, max_length=max_length, task_prompt=task_prompt)

        start_token_id = model.config.decoder_start_token_id
        print(f"task_prompt:{task_prompt}, start_token_id:{start_token_id}")
        encoder.eval()
        decoder.eval()

        image_data = Image.open(image_path).convert("RGB")
        save_torchscript, do_inference = False, True
        save_encoder_torchscript = "encoder_ts.pth"
        pixel_values = processor(image_data, return_tensor="pt").pixel_values
        # pixel_values = torch.randn(1, 3, 1024, 1024)

        pixel_values = torch.randn(1, 3, image_size[0], image_size[1])
        feature_len = int((image_size[0] / 32) * (image_size[1] / 32))

        print("========= step1.1 trace encoder! ==========")
        encoder_model = torch.jit.trace(encoder.to(device), pixel_values.to(device), strict=False)
        if save_torchscript:
            encoder_model.save(save_encoder_torchscript)

        encoder_output = encoder(pixel_values.to(device))["last_hidden_state"]
        print(f"========= step1.2 trace encoder finished! ========== \nencoder_output:{encoder_output}")

        print(f"========= step2 trace token_model! ==========")
        tokenModel = ChatTokenModel(decoder, max_length=max_length).to(device)
        # tokenModel =tokenModel.to(device)
        tokenModel.eval()
        # decoder input
        input_ids = torch.tensor([[start_token_id]], dtype=torch.int64).to(device)
        attention_mask = torch.tensor([[1]], dtype=torch.int).to(device)
        encoder_hidden_states = torch.randn(1, feature_len, 1024)  # 1024=(1024/32) * (1024/32)
        unfinshed_sequence = torch.tensor([1], dtype=torch.int).to(device)
        # cross attention
        past_key_values = [
            [
                torch.zeros([1, 16, 1, 64]).half().to(device),
                torch.zeros([1, 16, 1, 64]).half().to(device),
                torch.zeros([1, 16, feature_len, 64]).half().to(device),
                torch.zeros([1, 16, feature_len, 64]).half().to(device),
            ] for i in range(4)
        ]

        traced_token_model = torch.jit.trace(tokenModel.to(device),
                                             (input_ids.to(device),
                                              input_ids.to(device),
                                              encoder_hidden_states.to(device),
                                              attention_mask.to(device),
                                              unfinshed_sequence.to(device)
                                              )).to(device)
        if save_torchscript:
            traced_token_model.save("decoder_init.pth")
        print(f"========= step2.1 trace token_model_1 finished! ==========")

        # deocer model
        input_ids = torch.tensor([[57527]], dtype=torch.int64).to(device)
        attention_mask = torch.tensor([[1, 1]], dtype=torch.int).to(device)
        unfinshed_sequence = torch.tensor([1], dtype=torch.int).to(device)
        # cross attention
        past_key_values = [
            [
                torch.zeros([1, 16, 1, 64]).to(device),
                torch.zeros([1, 16, 1, 64]).to(device),
                torch.zeros([1, 16, feature_len, 64]).to(device),
                torch.zeros([1, 16, feature_len, 64]).to(device),
            ] for i in range(4)
        ]
        prepare_input_ids = input_ids[:, -1:]
        traced_token_model_2 = torch.jit.trace(tokenModel.to(device),
                                               (input_ids.to(device),
                                                prepare_input_ids.to(device),
                                                encoder_hidden_states.to(device),
                                                attention_mask.to(device),
                                                unfinshed_sequence.to(device),
                                                past_key_values)).to(device)
        if save_torchscript:
            traced_token_model_2.save("decoder_init.pth")
        print(f"========= step2.2 trace token_model_2 finished! ==========")

        """
            loopModel, 使用script方式导出
        """
        print(f"========= step3.1 trace loop model! ==========")
        input_ids = torch.tensor([[start_token_id]], dtype=torch.int64).to(device)
        attention_mask = torch.tensor([[1]], dtype=torch.int).to(device)
        encoder_hidden_states = encoder_output
        debug_mode = False
        if debug_mode:
            loopModel = ChatLoopModel(tokenModel, tokenModel).to(device)
            trace_output = loopModel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
            print(f"debug torch trace_ouptut:{trace_output}")

        loopModel = ChatLoopModel(traced_token_model, traced_token_model_2).to(device)
        # loopModel = tokenModel.to(device)
        trace_loop_model = torch.jit.script(loopModel).to(device)
        if save_torchscript:
            trace_loop_model.save("trace_loop_model.pth")
        print(f"========= step3.2 trace trace_loop_model finished! ==========")

        # 验证trace loopmodel
        trace_loop_model_output = loopModel(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            encoder_hidden_states=encoder_hidden_states.to(device)
        )
        print(f"========= step3.3 validate trace_loop_model_output: ==========\n{trace_loop_model_output}")
        print("".join(processor.tokenizer.convert_ids_to_tokens(trace_loop_model_output[0])))

        """
            总模型 trace
        """
        print(f"========= step4.1 trace chat model! ==========")
        model = ChatModel(trace_loop_model, encoder_model, decoder, tokenzier, max_length=max_length).to(device)
        model.to(device)
        model.eval()

        decoder_input_ids = processor.tokenizer(task_prompt,
                                                add_special_tokens=False,
                                                return_tensors="pt").input_ids.to(device)
        attention_mask = torch.tensor([[1]], dtype=torch.int).to(device)
        trace_model = torch.jit.trace(model.to(device),
                                      (pixel_values.to(device),
                                       decoder_input_ids.to(device),
                                       attention_mask.to(device)
                                       )).to(device)
        print(f"========= step4.1 trace chat model finish! ==========")
        trace_model.save(os.path.join(SAVE_MP, "model.pt"))
        print(f'========= step4.2 save model: {os.path.join(SAVE_MP, "model.pt")} ==========')

        print(f"========= step5 run export model test! ==========")
        for i in range(1):
            t1 = time.time()
            if device == "cuda":
                print(f"cuda memory: {torch.cuda.memory_allocated() / 1000 / 1000 / 1000}")
            output = trace_model(pixel_values.to(device), input_ids.to(device), attention_mask.to(device))
            print(f"output:{output}")
            print("".join(processor.tokenizer.convert_ids_to_tokens(output[0])))
            print(f"time {i} : {time.time() - t1}")

        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export-ts")
    parser.add_argument("--checkpoint_path", type=str, help="model checkpoint path")
    parser.add_argument("--save_torchscript_path", type=str, help="save_torchscript_path")
    parser.add_argument("--task_prompt", type=str, help="task_prompt")
    parser.add_argument("--image_size", type=list, help="image_size higth_width order")
    parser.add_argument("--test_image_path", type=str, help="test_image_path")
    parser.add_argument("--max_length", type=int, help="max_length")

    # export_torchscript(parser)
    export_torchscript()
