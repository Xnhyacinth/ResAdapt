

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 32

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 8

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 128

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 16 vid_list_mrope

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 8 vid_list_mrope

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 310 video_all 128 vid_list_mrope


# bash visionthink/scripts/eval.sh Video-R1/Qwen2.5-VL-7B-COT-SFT vllm_generate video_all 32 0 0 0 0 logs_eval vid_list_mrope

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval len 0

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval len 0

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval len 0

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval vid_list_mrope 0

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 64 0 0 0 0 logs_eval vid_list_mrope 0

bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 128 0 0 0 0 logs_eval vid_list_mrope 0


# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 8 0 0 0 0 logs_eval img 0

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval img 0

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 16 0 0 0 0 logs_eval img 0


# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32 0 0 0 0 logs_eval img_repad 0

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 16 0 0 0 0 logs_eval img_repad 0

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 8 0 0 0 0 logs_eval img_repad 0

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 32

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 16

# bash visionthink/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 8