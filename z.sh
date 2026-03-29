
bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.5

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.5

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.5


bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.1

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.25

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len_fix0.5

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len_fix0.5

bash visionthink/scripts/eval.sh Qwen/Qwen3-VL-8B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len_fix0.5


# accelerate launch --config_file /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ds2.yaml \
#   --num_processes 1 --main_process_port 12345 \
#   /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/train_predictor_coldstart.py \
#   --train_jsonl /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1_scales/train.jsonl \
#   --model_path /mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smol \
#   --save_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smolv2_sft_nll \
#   --batch_size 16 --epochs 5 --lr 1e-4 \
#   --loss_type mse --use_mean_prediction \
#   --mixed_precision bf16 \
#   --attn_implementation flash_attention_2 \
#   --save_every 1000 \
#   --max_frames 16 \
#   # --resume_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv2_sft/step_500


