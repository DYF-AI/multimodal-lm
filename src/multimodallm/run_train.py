import os
import random
import datetime
import datasets
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel, GenerationConfig, \
    Seq2SeqTrainingArguments
import pytorch_lightning as pl
from multimodallm.trainer import CustomDonutModelHFTrainer, CustomDonutModelPLTrainer, SaveModelCallback
from multimodallm.utils.data_utils import preprocess

os.environ["WANDB_API_KEY"] = "927e1a602dfef7063ed62c26589b2cd5f8dd189f"
os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    # 927e1a602dfef7063ed62c26589b2cd5f8dd189f
    date = datetime.datetime.now()
    all_train_config = {
        # 预训练
        "ocr_pretrain": {
            "MP": r"J:\model\pretrained-model\torch\donut-base-expand-vocab",
            "num_epoch":20,
            "max_length": 2560,
            "start_token": "<s_ocr_pretrain>",
            "image_size": [1024, 1024],
            "expand_vocab": ["<s_ocr_pretrain>", "</s_ocr_pretrain>", "<s_text_sequence>", "</s_text_sequence>"],
            "train_dataset": r"J:\data\mllm-data\mllm-pretrain-data\train.arrow",
            "validation_dataset":r"J:\data\mllm-data\mllm-pretrain-data\validation.arrow",
            "test_dataset":r"J:\data\mllm-data\mllm-pretrain-data\test.arrow",
            "model_save_path": os.path.join(r"J:\model\mllm-model\donut-pretrain", date.strftime('%Y%m%d')),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,  # huggingface 调用了共享GPU内存?
            "gradient_accumulation_steps": 4,
            "save_total_limit": 10,
            "learning_rate": 3e-5,
            "val_check_interval": 0.1,
            "seed": 2023,
            "val_num": 300,
            "patience": 20,
        },
        # 火车票
        "train_ticket": {
            "MP": r"J:\model\pretrained-model\torch\donut-base",
            #"MP": r"J:\model\mllm-model\train_ticket\checkpoint-760",
            "num_epoch": 20,
            "max_length": 768,
            "start_token": "<s_trainticket>",
            #"start_token": "<s_cord-v2>",
            "image_size": [1280, 960],
            "expand_vocab": ["<s_cord-v2>", "<s_train_ticket>", "</s_train_ticket>", "<s_starting_station>",
                             "</s_starting_station>","<s_destination_station>", "</s_destination_station>",
                             "<s_seat_category>", "</s_seat_category>","<s_ticket_rates>", "</s_ticket_rates>",
                             "<s_ticket_num>", "</s_ticket_num>","<s_date>", "</s_date>", "<s_train_num>", "</s_train_num>",
                             ],
            "train_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\train_cache.arrow",
            "validation_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\test_cache.arrow",
            "test_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\test_cache.arrow",
            "model_save_path": os.path.join(r"J:\model\mllm-model\train_ticket", date.strftime('%Y%m%d')),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1, # huggingface 调用了共享GPU内存?
            "gradient_accumulation_steps": 4, # 梯度累计速度会变慢很多？
            "save_total_limit": 5,
            "learning_rate": 3e-5,
            "val_check_interval": 0.5,#1.0,
            "seed": 2023,
            "val_num": 200,
            "patience": 5,
        },

    }

    task_name = "ocr_pretrain"#"train_ticket" #"ocr_pretrain"
    wandb_logger = WandbLogger(project="donut-model", name=task_name)
    random.seed(all_train_config[task_name]["seed"])
    # 扩展词表的donut-base
    #MP = r"J:\model\pretrained-model\torch\donut-base-expand-vocab"
    # 需要统计以下训练集的长度
    max_length = all_train_config[task_name]["max_length"]
    start_token = all_train_config[task_name]["start_token"]
    # height_width order [height, width], 宽高费用容易混淆
    image_size = all_train_config[task_name]["image_size"]
    config = VisionEncoderDecoderConfig.from_pretrained(all_train_config[task_name]["MP"])
    config.encoder.image_size = image_size  # (height, width)
    config.decoder.max_length = max_length

    expand_vocab = all_train_config[task_name]["expand_vocab"]
    ignore_mismatched_sizes = False
    if max_length > config.decoder.max_position_embeddings:
        config.decoder.max_position_embeddings = max_length
        ignore_mismatched_sizes = True

    processor = DonutProcessor.from_pretrained(all_train_config[task_name]["MP"])
    model = VisionEncoderDecoderModel.from_pretrained(all_train_config[task_name]["MP"], config=config,
                                                      ignore_mismatched_sizes=ignore_mismatched_sizes)

    newly_added_num = processor.tokenizer.add_tokens(expand_vocab)
    if newly_added_num > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([start_token])[0]
    print("Pad token ID:", processor.decode([model.config.pad_token_id]))
    print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

    processor.image_processor.size = image_size[::-1]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False

    train_dataset = all_train_config[task_name]["train_dataset"]
    validation_dataset = all_train_config[task_name]["validation_dataset"]
    test_dataset = all_train_config[task_name]["validation_dataset"]

    #train_key, validation_key, test_key = "train", "validation", "test"
    DP = r"J:\data\mllm-data\mllm-pretrain-data"
    en_ds = {}
    for dataset_path in [train_dataset, validation_dataset, test_dataset]:
        dataset = datasets.Dataset.from_file(dataset_path)#.select(range(10))
        key = os.path.basename(dataset_path).split(".")[0]
        if "_cache" in key: key = key.replace("_cache", "")
        # "eager":True, 缓存图片数据到内存，但是非常耗内存
        en_ds[key] = dataset.map(preprocess,
                                 fn_kwargs={"processor": processor,
                                            "sort_key": False,
                                            "eager":False,
                                            "random_padding": key == "train",
                                            "max_length": max_length},
                                 batched=True,
                                 batch_size=64,
                                 keep_in_memory=True)
    print(en_ds)
    print(en_ds["train"][0])


    use_huggingface_trainer = False
    """
        lighting-trainer在evalution的速度远远快于huggingface-trainer的速度, 原因带排查
        初步原因：huggingface有计算val_loss, 进行了两次forward? lightning只是算val_metric?
    """
    if use_huggingface_trainer:
        generation_config = GenerationConfig(
            max_length=max_length,
            early_stopping=True,
            num_beams=1,
            decoder_start_token_id=model.config.decoder_start_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]]
        )

        model_save_path = all_train_config[task_name]["model_save_path"]
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        training_args = Seq2SeqTrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=all_train_config[task_name]["num_epoch"],
            evaluation_strategy="epoch",
            dataloader_num_workers=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=all_train_config[task_name]["learning_rate"],
            per_device_train_batch_size=all_train_config[task_name]["per_device_train_batch_size"],
            per_device_eval_batch_size=all_train_config[task_name]["per_device_eval_batch_size"],
            gradient_accumulation_steps=all_train_config[task_name]["gradient_accumulation_steps"],
            remove_unused_columns=False,
            fp16=True,
            generation_config=generation_config,
            predict_with_generate=True,
            save_total_limit=all_train_config[task_name]["save_total_limit"],
            metric_for_best_model="norm_edit",
            report_to="none",
            run_name=task_name,
            dataloader_pin_memory=True,
        )

        custom_model = CustomDonutModelHFTrainer(
            model_config=config,
            training_args=training_args,
            processor=processor,
            model=model,
            expand_vocab=expand_vocab,
            train_dataset=en_ds["train"],
            eval_dataset=en_ds["validation"] if "validation" in en_ds else en_ds["test"]
        )
        custom_model.trainer.train()

    else:
        def custom_collate(batch, random_padding):
            ids, images, angles, ground_truths, labels, targets, random_paddings = [], [], [], [], [], [], []
            for sample in batch:
                ids.append(sample["id"])
                images.append(sample["image"].convert("RGB"))
                angles.append(sample["angle"])
                ground_truths.append(sample["ground_truth"])
                labels.append(sample["labels"])
                targets.append(sample["target"])
                random_paddings.append(sample["random_padding"])
            pixel_values = processor(images, random_padding=random_padding,
                                     return_tensors="pt").pixel_values#.squeeze()
            return ids, pixel_values, angles, ground_truths, torch.tensor(labels), targets, random_paddings

        val_num = all_train_config[task_name]["val_num"]
        en_val_dataset = en_ds["validation"].select(random.sample(range(len(en_ds["validation"])),  val_num)) \
            if "validation" in en_ds else en_ds["test"].select(random.sample(range(len(en_ds["test"])), val_num))
        train_dataloader = DataLoader(en_ds["train"],
                                      batch_size=all_train_config[task_name]["per_device_train_batch_size"],
                                      shuffle=True,
                                      collate_fn=lambda x: custom_collate(x, True),
                                      num_workers=0)
        val_dataloader = DataLoader(en_val_dataset,
                                    batch_size=all_train_config[task_name]["per_device_eval_batch_size"],
                                    shuffle=False,
                                    collate_fn=lambda x: custom_collate(x, False ),
                                    num_workers=0)

        batch = next(iter(train_dataloader))
        ids, images, angles, ground_truths, labels, targets, _ = batch

        lightning_config = {
            "num_epoch": all_train_config[task_name]["num_epoch"],
            "val_check_interval": all_train_config[task_name]["val_check_interval"], #1.0,  # 多少次进行验证, int: step, float:训练集的百分比进度（1.0，完全训练一个epcoh）
            "check_val_every_n_epoch": 1,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": all_train_config[task_name]["gradient_accumulation_steps"],
            "num_training_samples_per_epoch": 800,
            "lr": all_train_config[task_name]["learning_rate"],
            "train_batch_sizes": [all_train_config[task_name]["per_device_train_batch_size"]],
            "val_batch_sizes": [all_train_config[task_name]["per_device_eval_batch_size"]],
            "seed":all_train_config[task_name]["seed"],
            "num_nodes": 1,
            "warmup_steps": 300,  # 800/8*30/10, 10%
            "result_path": "./result",
            "verbose": True,
        }

        model_module = CustomDonutModelPLTrainer(lightning_config,
                                                 processor,
                                                 model,
                                                 train_dataloader,
                                                 val_dataloader,
                                                 max_length=all_train_config[task_name]["max_length"]
                                                 )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=lightning_config.get("max_epochs"),
            val_check_interval=lightning_config.get("val_check_interval"),
            check_val_every_n_epoch=lightning_config.get("check_val_every_n_epoch"),
            accumulate_grad_batches = lightning_config.get("accumulate_grad_batches"),
            gradient_clip_val=lightning_config.get("gradient_clip_val"),
            precision=16,  # we'll use mixed precision
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[
                EarlyStopping(monitor="val_metric",
                              patience=all_train_config[task_name]["patience"],
                              mode="max"
                              ),
                SaveModelCallback(all_train_config[task_name]["model_save_path"]),  # hf model save to local
            ],
        )
        trainer.fit(model_module)
        print("trained")