import os
import difflib
import json
from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple, List, Any
import re
import pytorch_lightning as pl
import datasets
import numpy as np
import torch
import yaml
import torch.nn as nn
from nltk import edit_distance
from pytorch_lightning import Callback
from transformers import GenerationConfig, Seq2SeqTrainingArguments, DonutProcessor, VisionEncoderDecoderConfig, \
    VisionEncoderDecoderModel, Seq2SeqTrainer, EarlyStoppingCallback
from mllm.collators import DataCollatorForGeneration
from mllm.utils.data_utils import convert_json_key_to_id, token2json, preprocess


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
    """
        huggingface trainer
    """

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
        # self.tokenizer = self.processor.tokenizer,
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
            # seq = re.sub(r"<.*?>", "", seq, count=1).strip() # remove first task start token, label是没有start_token的
            seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
            answer_sequence.append(seq)

        scores = []
        for pred, answer in zip(predict_sequences, answer_sequence):
            print(f"pred  :{self.processor.token2json(pred)}")
            print(f"answer:{self.processor.token2json(answer)}")
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


class SaveModelCallback(Callback):
    def __init__(self,
                 save_model_path,
                 save_all_validation_ckpt=True):
        self.best_val_metric = -100
        self.model_model_path = save_model_path
        self.save_all_validation_ckpt = save_all_validation_ckpt

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        print(f"save mode: {self.save_all_validation_ckpt}")
        model_save_path = f"{self.model_model_path}/pl-checkpoint-{trainer.global_step}_ned_{trainer.callback_metrics['val_metric']}"
        if trainer.callback_metrics['val_metric'] > self.best_val_metric:
            print(
                f"save current best model: epoch_{trainer.current_epoch}-ned-{trainer.callback_metrics['val_metric']}")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            pl_module.processor.save_pretrained(model_save_path,
                                                commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            pl_module.model.save_pretrained(model_save_path,
                                            commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            self.best_val_metric = trainer.callback_metrics['val_metric']
        elif self.save_all_validation_ckpt:
            print(
                f"save current best model: epoch_{trainer.current_epoch}-ned-{trainer.callback_metrics['val_metric']}")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            pl_module.processor.save_pretrained(model_save_path,
                                                commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            pl_module.model.save_pretrained(model_save_path,
                                            commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            self.best_val_metric = trainer.callback_metrics['val_metric']

    def on_train_end(self, trainer, pl_module):
        """训练结束前,再保存一次"""
        print(f"\nPushing model to the hub after training")
        # pl_module.processor.push_to_hub("nielsr/donut-demo", commit_message=f"Training done")
        # model_save_path = f"{self.model_model_path}/epoch_{trainer.current_epoch}_ned_{trainer.callback_metrics['val_metric']}"
        model_save_path = f"{self.model_model_path}/pl-checkpoint-{trainer.global_step}-ned-{trainer.callback_metrics['val_metric']}"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        pl_module.processor.save_pretrained(model_save_path, commit_message=f"Training done")
        pl_module.model.save_pretrained(model_save_path, commit_message=f"Training done")

    # def on_validation_epoch_end(self, trainer, pl_module):
    def on_validation_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        print(f"save mode: {self.save_all_validation_ckpt}")
        model_save_path = f"{self.model_model_path}/pl-checkpoint-{trainer.global_step}-ned-{trainer.callback_metrics['val_metric']}"
        if trainer.callback_metrics['val_metric'] > self.best_val_metric:
            print(
                f"\nsave model: {trainer.callback_metrics['val_metric']} , better than best_val_metric: {self.best_val_metric}")
            print(f"Pushing model to the hub after validation")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            pl_module.processor.save_pretrained(model_save_path, commit_message=f"validation done")
            pl_module.model.save_pretrained(model_save_path, commit_message=f"validation done")
            self.best_val_metric = trainer.callback_metrics['avg_val_metric']
        elif self.save_all_validation_ckpt:
            print(
                f"\nsave model: {trainer.callback_metrics['val_metric']} , better than best_val_metric: {self.best_val_metric}")
            print(f"Pushing model to the hub after validation")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            pl_module.processor.save_pretrained(model_save_path, commit_message=f"validation done")
            pl_module.model.save_pretrained(model_save_path, commit_message=f"validation done")
            self.best_val_metric = trainer.callback_metrics['avg_val_metric']


class CustomDonutModelPLTrainer(pl.LightningModule):
    """
        pytorch-lightning trainer
    """

    def __init__(self,
                 config,
                 processor,
                 model,
                 model_train_dataloader,
                 model_val_dataloader,
                 max_length=768):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.model_train_dataloader = model_train_dataloader
        self.model_val_dataloader = model_val_dataloader
        self.max_length = max_length
        self.validation_step_outputs = []
        self.total_train_loss = 0
        self.total_train_step = 0
        self.val_metric_in_validation = 0
        self.val_metric_nums = 0

    def training_step(self, batch, batch_idx):
        ids, pixel_values, angles, ground_truths, labels, targets, _ = batch
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        # 如果不加item(),内存空间会越来越小
        # RuntimeError: [enforce fail at C:\cb\pytorch_1000000000000\work\c10\core\impl\alloc_cpu.cpp:81] data. DefaultCPUAllocator: not enough memory: you tried to allocate 4194304 bytes.
        self.total_train_loss += loss.item()
        self.total_train_step += 1
        self.log_dict({"train_loss": loss, "avg_train_loss": self.total_train_loss / (self.total_train_step + 0.001)},
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        ids, pixel_values, angles, ground_truths, labels, targets, _ = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        outputs = self.model.generate(pixel_values,
                                      decoder_input_ids=decoder_input_ids,
                                      max_length=self.max_length,
                                      early_stopping=True,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True, )
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = list()
        for id, pred, answer in zip(ids, predictions, ground_truths):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            pred = json.dumps(self.processor.token2json(pred), ensure_ascii=False)  # ground_truths是json
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            norm_ed = 1 - (edit_distance(pred, answer) / max(len(pred), len(answer)))
            scores.append(norm_ed)
            self.val_metric_in_validation += norm_ed
            self.val_metric_nums += 1
            # self.log_dict({"avg_val_metric": self.val_metric_in_validation/(self.val_metric_nums)}, sync_dist=True)
            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"\nid:{id}")
                print(f"Prediction: {self.processor.token2json(pred)}")
                print(f"Answer: {self.processor.token2json(answer)}")
                print(f" Normed ED: {scores[0]}")
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            self.validation_step_outputs = [self.validation_step_outputs]
        assert len(self.validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        self.log_dict({"avg_val_metric": self.val_metric_in_validation / (self.val_metric_nums)}, sync_dist=True)
        self.validation_step_outputs.clear()
        self.val_metric_in_validation = 0
        self.val_metric_nums = 0

    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return self.model_train_dataloader

    def val_dataloader(self):
        return self.model_val_dataloader
