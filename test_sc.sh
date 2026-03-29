
# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8 20 600 video_all 32

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8 20 600 video_all 32 vid_list_mrope

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8 20 600 video_all 128

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8 20 600 video_all 128 vid_list_mrope


# # bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 32

# # bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 32 base_mrope

# # bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 16 base_mrope

# # bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 8 base_mrope


bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 32 0 0 0 0 logs_eval len

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 32 0 0 0 0 logs_eval vid_list_mrope

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 64 0 0 0 0 logs_eval vid_list_mrope

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 64 0 0 0 0 logs_eval len

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 128 0 0 0 0 logs_eval vid_list_mrope

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 128 0 0 0 0 logs_eval len



bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 32 0 0 0 0 logs_eval vid_list_mrope_sys

# bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 32 0 0 0 0 logs_eval sys

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 64 0 0 0 0 logs_eval vid_list_mrope_sys

# bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 64 0 0 0 0 logs_eval vid_list_mrope

bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 128 0 0 0 0 logs_eval vid_list_mrope_sys

# bash visionthink/scripts/eval.sh Video-R1/Video-R1-7B vllm_generate video_all 128 0 0 0 0 logs_eval vid_list_mrope