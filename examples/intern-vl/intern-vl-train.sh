# Experimental environment: A100
# 80GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --max_length 4096

## device_map
## Experimental environment: 2*A100...
## 2*43GB GPU memory
#CUDA_VISIBLE_DEVICES=0,1 swift sft \
#    --model_type  internvl-chat-v1_5 \
#    --dataset coco-en-2-mini \
#    --max_length 4096
#
## ddp + deepspeed-zero2
## Experimental environment: 2*A100...
## 2*80GB GPU memory
#NPROC_PER_NODE=2 \
#CUDA_VISIBLE_DEVICES=0,1 swift sft \
#    --model_type  internvl-chat-v1_5 \
#    --dataset coco-en-2-mini \
#    --max_length 4096 \
#    --deepspeed default-zero2


## Experimental environment: 4 * A100
## device map
## 4 * 72 GPU memory
#CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
#    --model_type internvl-chat-v1_5 \
#    --dataset coco-en-2-mini \
#    --max_length 4096 \
#    --sft_type full \