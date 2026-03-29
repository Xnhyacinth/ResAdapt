# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 16

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 8

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 16 base_mrope

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 8 base_mrope

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 128

# bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 110 120 video_all 128 base_mrope


# bash visionthink/scripts/eval.sh IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B qwen2_5_vl_autothink video 32 0 0 0 0 logs_eval 0

# bash visionthink/scripts/eval.sh IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B vllm_generate_autothink video_all 32 0 0 0 0 logs_eval sys

# bash visionthink/scripts/eval.sh IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B qwen2_5_vl_autothink video_all 32 0 0 0 0 logs_eval sys

bash visionthink/scripts/eval.sh IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B qwen2_5_vl_autothink video_all 64 0 0 0 0 logs_eval sys

bash visionthink/scripts/eval.sh IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B qwen2_5_vl_autothink video_all 128 0 0 0 0 logs_eval sys
