import difflib
import os
from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple, List, Any
import re

import datasets
import numpy as np
import torch
import yaml
import torch.nn as nn
from transformers import GenerationConfig, Seq2SeqTrainingArguments, DonutProcessor, VisionEncoderDecoderConfig, \
    VisionEncoderDecoderModel, Seq2SeqTrainer, EarlyStoppingCallback
from multimodallm.collator import DataCollatorForGeneration
from multimodallm.utils.data_utils import convert_json_key_to_id, token2json, preprocess

class GenSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if self.model.generation_config is not None:
            for k, v in self.model.generation_config.to_diff_dict().items():
                if k in gen_kwargs and gen_kwargs[k] is not None:
                    continue
                gen_kwargs[k] = v
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)


class CustomDonutModelHFTrainer:
    def __init__(self,
                 model_config: VisionEncoderDecoderConfig,
                 training_args: Seq2SeqTrainingArguments,
                 processor: DonutProcessor,
                 model: VisionEncoderDecoderModel,
                 expand_vocab=None,
                 train_dataset: datasets.Dataset = None,
                 eval_dataset: datasets.Dataset = None):
        if expand_vocab is None:
            expand_vocab = ["<s_ocr_pretrain>", "</s_ocr_pretrain>", "<text_sequence>, </text_seqence>"]
        self.model_config = model_config
        self.training_args = training_args
        self.processor = processor
        self.model = model
        #self.tokenizer = self.processor.tokenizer,
        self.expand_vocab = expand_vocab
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self._init_data_collator()
        self.key_ids = self._init_key_ids()
        self._init_trainer()

    def _init_data_collator(self):
        self._data_collator = DataCollatorForGeneration(
            processor=self.processor,
            tokenizer=self.processor.tokenizer,
            padding="max_length",
            max_length=self.model_config.decoder.max_length
        )

    def _init_key_ids(self):
        key_ids = dict()
        for v in self.expand_vocab:
            if v.startswith("</") or v.endswith("/>"):
                continue
            key = v.replace("<s_", "").replace(">", "")
            key_ids[key] = len(key_ids)
        return key_ids

    def computer_metrics(self, eval_pred):
        prediction, labels = eval_pred
        pred_decode_list = self.processor.tokenizer.batch_decode(prediction)

        predict_sequences = list()
        for seq in pred_decode_list:
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
            predict_sequences.append(seq)
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        label_decode_list = self.processor.tokenizer.batch_decode(labels)
        answer_sequence = list()
        for seq in label_decode_list:
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            #seq = re.sub(r"<.*?>", "", seq, count=1).strip() # remove first task start token, label是没有start_token的
            seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
            answer_sequence.append(seq)

        scores = []
        for pred, answer in zip(predict_sequences, answer_sequence):
            #print(f"pred json:{token2json(pred, expand_vocab=self.expand_vocab)}")
            #print(f"answer json:{token2json(answer, expand_vocab=self.expand_vocab)}")
            print(f"pred json:{self.processor.token2json(pred)}")
            print(f"answer json:{self.processor.token2json(answer)}")
            answer = convert_json_key_to_id(token2json(answer, expand_vocab=self.expand_vocab), key_ids=self.key_ids)
            # answer = token2json(answer)
            answer = yaml.dump(answer, allow_unicode=True, sort_keys=False)
            pred = convert_json_key_to_id(token2json(pred, expand_vocab=self.expand_vocab), key_ids=self.key_ids)
            pred = yaml.dump(pred, allow_unicode=True, sort_keys=False)
            score = difflib.SequenceMatcher(None, pred, answer).quick_ratio()
            scores.append(score)

        return {
            "norm_edit": np.mean(scores),
        }

    def _init_trainer(self):
        self.trainer = GenSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self.computer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
