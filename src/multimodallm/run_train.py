import os

import datasets
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel, GenerationConfig, \
    Seq2SeqTrainingArguments

from multimodallm.trainer import CustomDonutModelHFTrainer
from multimodallm.utils.data_utils import preprocess

if __name__ == "__main__":

    all_train_config = {
        # 预训练
        "ocr_pretrain": {
            "MP": r"J:\model\pretrained-model\torch\donut-base-expand-vocab",
            "max_length": 2560,
            "start_token": "<s_ocr_pretrain>",
            "image_size": [1024, 1024],
            "expand_vocab": ["<s_ocr_pretrain>", "</s_ocr_pretrain>", "<s_text_sequence>", "</s_text_sequence>"],
            "train_dataset": r"J:\data\mllm-data\mllm-pretrain-data\train.arrow",
            "validation_dataset":r"J:\data\mllm-data\mllm-pretrain-data\validation.arrow",
            "test_dataset":r"J:\data\mllm-data\mllm-pretrain-data\test.arrow",
            "model_save_path": r"J:\model\mllm-model\donut-pretrain",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "save_total_limit": 5,
        },
        # 火车票
        "train_ticket": {
            #"MP": r"J:\model\pretrained-model\torch\donut-base",
            "MP": r"J:\model\mllm-model\train_ticket\checkpoint-760",
            "max_length": 768,
            #"start_token": "<s_trainticket>",
            "start_token": "<s_cord-v2>",
            "image_size": [1280, 960],
            "expand_vocab": ["<s_cord-v2>", "<s_train_ticket>", "</s_train_ticket>", "<s_starting_station>", "</s_starting_station>",
                             "<s_destination_station>", "</s_destination_station>", "<s_seat_category>", "</s_seat_category>",
                             "<s_ticket_rates>", "</s_ticket_rates>", "<s_ticket_num>", "</s_ticket_num>",
                             "<s_date>", "</s_date>", "<s_train_num>", "</s_train_num>",
                             ],
            "train_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\train.arrow",
            "validation_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\test.arrow",
            "test_dataset": r"J:\data\mllm-data\mllm-finetune-data\trainticket\test.arrow",
            "model_save_path": r"J:\model\mllm-model\train_ticket",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1, # 梯度累计速度会变慢很多？
            "save_total_limit": 5,
        },

    }

    task_name = f"train_ticket"

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
        en_ds[key] = dataset.map(preprocess,
                                 fn_kwargs={"processor": processor, "sort_key": False, "random_padding": key == "train",
                                            "max_length": max_length}, batched=True, batch_size=16, keep_in_memory=True)
    print(en_ds)

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
        num_train_epochs=20,
        evaluation_strategy="epoch",
        dataloader_num_workers=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=3e-5,
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
        train_dataset=en_ds["train"].select(range(10)),
        eval_dataset=en_ds["validation"] if "validation" in en_ds else en_ds["test"].select(range(30))
    )
    custom_model.trainer.train()
