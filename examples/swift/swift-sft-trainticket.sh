
#CUDA_VISIBLE_DEVICES=0 swift sft \
#  --model_type internvl2-1b \
#  --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B \
#  --output_dir /mnt/n/model/sft-model/trainticket-sft \
#  --dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_train.jsonl \
#  --val_dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_val.jsonl \
#  --max_length 1024 \
#  --gradient_accumulation_steps 8 \
#  --eval_steps 200 \
#  --save_steps 200 \
#  --num_train_epochs 10 \
#  --batch_size 1 \
#  --learning_rate 3e-4 \
#  --lr_scheduler_type cosine \
#  --sft_type lora \
#  --lora_target_modules ALL \
#  --weight_decay 0.1 \
#  --max_grad_norm 0.5 \
#  --dtype fp16 \
#  --lora_dtype fp16 \
#  --lora_rank 8 \
#  --lora_alpha 32 \
#  --lora_dropout 0.05 \
#  --do_sample False \
#  --warmup_ratio 0.05 \
#
##  --gradient_checkpointing True \
##  --predict_with_generate True \



CUDA_VISIBLE_DEVICES=0 swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path /mnt/n/model/Qwen/Qwen2-VL-7B-Instruct \
  --output_dir /mnt/n/model/sft-model/trainticket-sft \
  --dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_train.jsonl \
  --val_dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_val.jsonl \
  --max_length 1024 \
  --gradient_accumulation_steps 8 \
  --eval_steps 200 \
  --save_steps 200 \
  --num_train_epochs 10 \
  --batch_size 1 \
  --learning_rate 3e-4 \
  --lr_scheduler_type cosine \
  --sft_type lora \
  --lora_target_modules ALL \
  --weight_decay 0.1 \
  --max_grad_norm 0.5 \
  --dtype fp16 \
  --lora_dtype fp16 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --do_sample False \
  --warmup_ratio 0.05 \
  --predict_with_generate True \


#  --gradient_checkpointing True \
#  --predict_with_generate True \