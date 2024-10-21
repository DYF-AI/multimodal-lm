export PYTHONPATH=../../../

CUDA_VISIBLE_DEVICES=0 python got_ocr2_infer.py \
  --model_path /mnt/n/model/ocr-pretrain-model/GOT-OCR2_0 \
  --ocr_type format \
  --image_folder /mnt/n/data/mllm-data/mllm-pretrain-data/test/image-book \
  --save_res_path /mnt/n/data/mllm-data/mllm-pretrain-data/swift_label/metadata_image_book_test.jsonl
