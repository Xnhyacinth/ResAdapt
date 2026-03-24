
nohup bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 0 0 0 0 0 logs_eval fix0.5 > logs/fix0.5.log 2>&1 &

nohup bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate images 0 0 0 0 0 logs_eval fix1.5 > logs/fix1.5.log 2>&1 &

bash eval.sh YOUR_WORKSPACE_PATH/resadapt/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 250 290 video_add 1 all

bash eval.sh YOUR_WORKSPACE_PATH/resadapt/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 250 290 video_all 1 base_mrope_qwen3vl_8B


NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv4_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 2e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv4_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_2e-5.log 2>&1 &


### qwen3-vl-4b
NNODES=2 nohup bash resadapt/scripts/main.sh 7B mix_base_nframes64_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes64_video_7b_n16_mini64_bsz128_len8_resp8.log 2>&1 &


NNODES=2 MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct nohup bash resadapt/scripts/main.sh 4B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes32_video_4b_n16_mini64_bsz128_len4_resp8.log 2>&1 &

