
import sys

ms_swift_dir = "/mnt/g/dongyongfei786/custom-swift"
sys.path.append(ms_swift_dir)

from swift.llm import sft_main

if __name__ == '__main__':
    output = sft_main()

# --model /mnt/n/model/Qwen/Qwen2-VL-2B-Instruct --output_dir /mnt/n/model/sft-model/refcoco-sft --train_type full --dataset /mnt/n/data/coco_2014_grounding/train_grounding.jsonl --val_dataset /mnt/n/data/coco_2014_grounding/validation_grounding.jsonl --max_length 1024 --gradient_accumulation_steps 8 --eval_steps 200 --save_steps 200 --num_train_epochs 10 --learning_rate 3e-4 --lr_scheduler_type cosine --target_modules all-linear --predict_with_generate True