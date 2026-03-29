# python mix_data.py \
#   --tspo4_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames \
#   --tspo10k_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K \
#   --general_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Train \
#   --out_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-Mixedpath \
#   # --disable_cache



# python3 preprocess_videodata_mix.py \
#   --local_dataset_root /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K \
#   --local_save_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data \
#   --val_size 500 \
#   --num_proc 16


# find /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data -type f -name "*.zip" | \
# xargs -n 1 -P 8 sh -c '
# zipfile="$0"
# dir=$(dirname "$zipfile")
# echo "Unzipping: $zipfile"
# unzip -o "$zipfile" -d "$dir"
# '

# python preprocess_videor1.py \
#   --local_dataset_path /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data/Video-R1-260k.json \
#   --local_dataset_root /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data \
#   --local_save_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1 \
#   --num_proc 16 \


# find /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/VideoAuto-R1-Data \
#   -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) \
#   | xargs -n 1 -P 16 -I {} bash -c '
#       echo "[UNPACK] {}"
#       dir=$(dirname "{}")
#       tar -xf "{}" -C "$dir"
#   '

# DAPO-Math,

# python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/preprocess_videoautor1.py \
#   --dataset_names VIRL,ThinkLite-VL-Hard,ActivityNet-TVG,Charades-STA,TimeR1,NeXT-GQA,VideoR1,TVBench,STI-Bench,MMR-VBench \
#   --dataset_config_path /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/VideoAuto-R1-Data/config.json \
#   --local_save_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1 \
#   --split train \
#   --rl_mode cot_rl


# python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/mix_data.py \
#   --videor1_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1/train.parquet \
#   --videoautor1_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1/train.parquet \
#   --output_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train.parquet \
#   --sample_size 16500 \
#   --problem_types ocr,free-form,regression \
#   --seed 42


# find logs_video -type f -name '*frames128*' | while read -r f; do
#     new="${f//frames128/frames8}"
#     echo "$f -> $new"
#     mv "$f" "$new"
# done


# python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/compute_frame_lengths.py \
#   --input_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train_with_frame_lengths1.parquet \
#   --output_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train_lens.parquet \
#   --model_path /mnt/bn/jiangzhongtao/users/liaohuanxuan/models/qwen2.5_vl-7B \
#   --frames 128


# python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/frame_sample.py \
#   --input_parquet /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train.parquet \
#   --output_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/samples \
#   --sample_size 10 \
#   --num_frames 16 \
#   --seed 42

# gpt-5.2-2025-12-11 gemini-3-flash-preview gpt-5-2025-08-07

export OPENAI_API_URL="https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
# export OPENAI_API_URL="https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
export OPENAI_API_KEY="dxMlgIJpXgkdou8z77OKt5rg4BQjwgJZ_GPT_AK"
python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/prepare_videor1_cot_scales1.py \
  --input_json /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data/Video-R1-COT-165k.json \
  --data_root /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data \
  --output_jsonl /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1_scales/trainv3.jsonl \
  --api_base_url https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi \
  --model gpt-5.2-2025-12-11 \
  --num_frames 32 \
  --fps 2.0 \
  --max_concurrency 128 \
  --retry 8 \
  --retry_wait 3.0 \
  --retry_jitter 0.3



# python3 /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/offline_scales_vllm.py \
#   --input_json /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data/Video-R1-COT-165k.json \
#   --data_root /mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data \
#   --output_jsonl /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1_scales/vllm_offline.jsonl \
#   --model Qwen/Qwen2.5-VL-7B-Instruct \
#   --num_frames 32