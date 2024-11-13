
## 支持使用zero3进行微调
#CUDA_VISIBLE_DEVICES=0 swift sft \
#  --model_type qwen2-vl-2b-instruct \
#  --model_id_or_path /mnt/n/model/Qwen/Qwen2-VL-2B-Instruct \
#  --sft_type lora \
#  --dataset refcoco-unofficial-grounding#20000 \
#  --deepspeed default-zero3


CUDA_VISIBLE_DEVICES=0 swift sft \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /mnt/n/model/Qwen/Qwen2-VL-2B-Instruct \
  --output_dir /mnt/n/model/sft-model/refcoco-sft \
  --sft_type lora \
  --dataset /mnt/n/data/coco_2014_grounding/validation_grounding.jsonl \
  --val_dataset /mnt/n/data/coco_2014_grounding/validation_grounding.jsonl \
  --max_length 1024 \
  --gradient_accumulation_steps 8 \
  --eval_steps 200 \
  --save_steps 200 \
  --num_train_epochs 10 \