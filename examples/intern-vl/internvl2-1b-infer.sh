#export PYTHONPATH=../../

CUDA_VISIBLE_DEVICES=0 python /mnt/g/dongyongfei786/multimodal-lm/mllm/predict/internvl2/predict.py \
  --ckpt /mnt/n/model/sft-model/internvl2-1b-trainticket/internvl2-1b/v22-20241030-152458/checkpoint-950-merged \
  --val_dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_test.jsonl \
  --save_result_path /mnt/n/model/sft-model/internvl2-1b-trainticket/internvl2-1b/v22-20241030-152458/checkpoint-950-merged/infer_result/pred.jsonl
