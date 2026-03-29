

# bash visionthink/scripts/abase_remote.sh > logs/async_remote.log 2>&1 

# bash visionthink/scripts/base.sh > logs/sync_new_8.log 2>&1
# # sleep 0.6h
# bash visionthink/scripts/abase.sh > logs/a.log 2>&1 

# bash visionthink/scripts/base.sh > logs/b.log 2>&1 

# # ray start --head --port 8288 --dashboard-port 8289

# # uv run bash visionthink/scripts/run.sh > logs/a.log 2>&1 


# huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir /mnt/bn/jiangzhongtao/cailincan/base_models/Qwen3-Embedding-0.6B --local-dir-use-symlinks False


# huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir /mnt/bn/jiangzhongtao/cailincan/base_models/Qwen3-Embedding-4B --local-dir-use-symlinks False

# huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir /mnt/bn/jiangzhongtao/cailincan/base_models/Qwen3-Embedding-8B --local-dir-use-symlinks False

# huggingface-cli download BAAI/bge-m3 --local-dir /mnt/bn/jiangzhongtao/cailincan/base_models/bge-m3 --local-dir-use-symlinks False

# huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir /mnt/bn/jiangzhongtao/cailincan/base_models/bge-reranker-v2-m3 --local-dir-use-symlinks False

python preprocess_data.py --local_dataset_path "/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/TSPO-10K/TSPO_10k.jsonl" --local_save_dir "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames-path" \
    --save_images_to_disk --image_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames-images

python visionthink/mix_data.py \
  --tspo4_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames \
  --tspo10k_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K \
  --general_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Train \
  --out_dir /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-Mixed


serve run pred_serve:app --address 0.0.0.0:8898

ray serve run pred_serve.py --host 0.0.0.0 --port 8000

bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate_custom video 16 0 0 0 0 logs_eval img debug

nohup bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate images 0 0 0 0 0 logs_eval fix0.5 > logs/fix0.5.log 2>&1 &

nohup bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate images 0 0 0 0 0 logs_eval fix1.5 > logs/fix1.5.log 2>&1 &

bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate images 0 0 0 0 0 logs_eval fix1.5 debug

bash scripts/eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify/scale_flash_filter-fsdp2-s8-7B-Single-bsz128-mini32-n2-min0.2-max2.0-len4-resp8/global_step_100 vllm_generate images 0 0 0 0 0 logs_eval 0 debug

bash convert.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/verl/ckpts/GeneralQA_Qwen_Verify qwen2_5_vl > logs/conv0.log 2>&1

bash convert.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify qwen2_5_vl > logs/conv.log 2>&1

nohup bash convert.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify qwen2_5_vl > logs/conv1.log 2>&1 &

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 400 video_all 16

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 400 video_all 32 vid_list_mrope

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate base_nframes8_video-fsdp2-7B-bsz64-mini16-n16-len4-resp8 300 400 video_all 8 vid_list_mrope

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz64-mini16-n1-max2.0-len4-resp8_1e-6 200 400 video_all 32

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz64-mini16-n1-max2.0-len4-resp8_1e-6 200 400 video_all 32 base_mrope

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 100 400 video_all 32

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 100 400 video_all 32 base_mrope

nohup bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate scale_flash_filt_tie_enc0.2_cen0.5-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 360 400 > logs/scale_7b.log 2>&1 &

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate scale-4-3B

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts0/GeneralQA_Qwen_Verify vllm_generate scale_flash_filter-fsdp2-s8-7B-Single-bsz128-mini32-n2-min0.2-max2.0-len4-resp8 100 400

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate scale_flash_frozen_filter_cost0.1_enc0.02-fsdp2-s8

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate disc 0 400

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate ray2 120 120

nohup bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate len8-resp8 > logs/base_resp8.log 2>&1 &

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate scale_flash_filt_tie_enc0.1_cen0.5-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8 10 10

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify vllm_generate scale_fla_is_filt_tie_enc0.2_cen0.5-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6 100 400

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8 10 600 video_all 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_base_nf128_video-dp2-7B-bsz128-mini64-n16-l8-r8 10 600 video_all 32 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r2-mix_base_nf32_video-dp2-7B-bsz128-mini64-n16-l4-r8 30 600 all 1

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r2-mix_base_nf32_video-dp2-7B-bsz128-mini64-n16-l4-r8 70 70 video_all 1 vid_list_mrope

bash eval_sc.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r2-mix_base_nf32_video-dp2-7B-bsz128-mini64-n16-l4-r8 70 70 video_all 1 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r2-mix_base_nf32_video-dp2-7B-bsz128-mini64-n16-l4-r8 10 600 video_all 1 vid_list_mrope_sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r2-mix_base_nf32_video-dp2-7B-bsz128-mini64-n16-l4-r8 10 600 video_all 1 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 10 600 video_all 1

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 10 600 video_all 128 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 10 600 video_all 1 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 10 600 video_all 1 base_mrope_sys 

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_sc_gt_filt_tie_enc0.2_cen0.5_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 10 600 video_all 32 base_mrope

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_sc_filt_tie_enc0.2_cen0.5_nf8_vid_mrope-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_5e-6 30 600 video_all 32 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_sft_hadw_sc_is_filt_new_enc0.2_nf128_vid_mrope-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_5e-6 10 600 video_all 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz32-mini16-n1-max2.0-l4-r8_5e-6 10 600 video_all 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_hadw_sc_gt_is_filt_new_enc0.2_nf32_vidm-dp2-s16-7B-bsz32-mini16-n1-max2.0-l4-r8_5e-6 10 600 video_all 32 sys

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_sc_gt_filt_tie_enc0.2_cen0.5_nf32_vidm-dp2-s16-7B-bsz32-mini16-n1-max2.0-l4-r8_5e-6 10 600 video_all 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_s_sc_gt_is_filt_fr_new_enc0.1_cen0.3_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_2e-6 10 600 video_all 32 base_mrope

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_s_sc_gt_is_filt_fr_new_enc0.1_cen0.3_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_2e-6 10 600 video_all 32 base_mrope

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awa_scv1_sim_filt_fr_acc_cost_enc0.15_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_2e-6 10 600 video 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awa_scv2_sim_filt_fr_acc_cost_enc0.15_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_2e-6 10 600 video 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_sc_gt_filt_tie_enc0.2_cen0.5_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-6 90 130 video 32 base_mrope

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awa_scv2bd_sim_filt_fr_acc_cost_enc0.25_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_2e-4 40 600 video 32

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_scv3b_sim_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 40 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_scv2bd_sim_filt_fr_new_enc0.15_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 40 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_scv3bd0_sim_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 40 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smol_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 40 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv2_sim_ent_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 40 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smol_simerror_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awrk_swi_scv3bd0_sim_filt_fr_acc_cost_enc0.25_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv2_sim1_ent_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smol_sim1_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_1e-4 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awrk_swi_scv3bd0_sim1_filt_fr_acc_cost_enc0.25_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv1_sim_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_awid_swi_sc_smolv1_sim_ccen_filt_fr_acc_cost_enc0.25_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolhead_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolembed_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 20 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf128_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_2e-5 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_sft_asym_sc_smolv1_sim_filt_fr_new_enc0.25_cen0.35_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf64_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_5e-5 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf64_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_5e-5 30 600 video 1 all

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf128_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_2e-5 30 600 video 1 base_mrope_qwen3vl_8B

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf64_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_5e-5 30 600 video 1 base_mrope_qwen3vl_8B

bash visionthink/predictor/main.sh 3B scale_flash_filter_tie_cost_enc0.02 4 8 fsdp2 2.0 2 8 1e-5 debug

bash visionthink/predictor/main.sh 3B base 4 4 fsdp2 2.5 8 0 1e-5 debug

bash visionthink/predictor/main.sh 7B scale_flash_filter 8 8 fsdp2 1.5 1 16 1e-5 8 images debug

bash visionthink/predictor/main.sh 7B base 8 8 fsdp2 1.5 16 1 1e-5 8 0 debug

bash visionthink/predictor/main.sh 7B scale_flash_filter_nframes8 8 8 fsdp2 1.5 1 16 1e-5 debug

bash visionthink/predictor/main.sh 7B base_nframes8_video 8 8 fsdp2 1.5 16 1 1e-5 debug

bash visionthink/predictor/main.sh 7B base_nframes8_vid_list 8 8 fsdp2 1.5 16 1 1e-5 debug

bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.1_nframes8_video_mrope 8 8 fsdp2 1.5 1 16 1e-5 debug

# video

## base

nohup bash visionthink/predictor/main.sh 7B base_nframes8_offline 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes8_offline_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes8 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes8_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes8_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes8_video_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes8_vid_list 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes8_vid_list_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes16_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes16_video_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes4 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes4_7b_n16_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base_nframes8_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/base_nframes8_video_7b_n16_mini16_bsz64_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B r1_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/r1_base_nframes32_video_7b_n16_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_base_nframes32_video_7b_n16_mini16_bsz64_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_base_nframes32_video_7b_n16_mini16_bsz32_len4_resp8.log 2>&1 &

## scale

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_nframes8_offline 8 8 fsdp2 1.5 1 16 1e-5 0 > logs_run/scale_flash_filter_nframes8_offline_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8.log 2>&1 &

# nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.5_nframes8_offline 8 8 fsdp2 1.5 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.1_cen0.5_nframes8_offline_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8.log 2>&1 &

# nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_nframes8 8 8 fsdp2 1.5 1 16 1e-5 0 > logs_run/scale_flash_filter_nframes8_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8 8 8 fsdp2 1.5 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.25_cen0.25_nframes8 8 8 fsdp2 1.5 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.25_cen0.25_nframes8_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8.log 2>&1 &

# nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_tie_acc_cost_enc0.15_cen0.4_nframes8_offline 8 8 fsdp2 1.5 1 16 1e-4 0 > logs_run/scale_flash_forzen_filter_tie_acc_cost_enc0.15_cen0.4_nframes8_offline_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8_lr1e-4.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes4 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes4_fsdp2_7b_s16_n1_min0.2_max_2.0_mini8_bsz32_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_vid_list 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_vid_list_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.25_nframes8_vid_list 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.1_cen0.25_nframes8_vid_list_fsdp2_7b_s16_n1_min0.25_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_vid_list 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_vid_list_fsdp2_7b_s16_n1_min0.2_max_2.0_mini8_bsz32_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/scale_flash_filter_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.25_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

WANDB_RUN_ID=run_20260107_2df90221 nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_5e-6.log 2>&1 &

WANDB_RUN_ID=run_20260109_7d354c7c nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.1_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_5e-6.log 2>&1 &

# WANDB_RUN_ID=run_20260109_4eb43704 nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.1_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/scale_flash_filter_tie_acc_cost_enc0.2_cen0.1_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_5e-6.log 2>&1 &

# WANDB_RUN_ID=run_20260108_b8eaeab8 nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_2e-6.log 2>&1 &

WANDB_RUN_ID=run_20260108_b44f2290 nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-5.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_notest 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_notest_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.2_cen0.5_bn_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.2_cen0.5_bn_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_gdpo_acc_cost_enc0.2_bn_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/scale_flash_ispred_filter_gdpo_acc_cost_enc0.2_bn_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-5.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_mix 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_mix_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_newtie_acc_cost_enc0.2_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/scale_flash_ispred_filter_newtie_acc_cost_enc0.15_nframes8_video_mrope_mix_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B r1_scale_flash_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/r1_scale_flash_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B r1_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/r1_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B mix_scale_flash_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_scale_flash_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz32_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B mix_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz32_len4_resp8_5e-6.log 2>&1 &

## ray

NNODES=4 nohup bash visionthink/predictor/main.sh 7B r1_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/r1_ray4_base_nframes32_video_7b_n16_mini32_bsz128_len4_resp8.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B r1_base_nframes128_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/r1_ray4_base_nframes128_video_7b_n16_mini32_bsz128_len8_resp8.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes128_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray4_base_nframes128_video_7b_n16_mini64_bsz128_len8_resp8.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes64_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray4_base_nframes64_video_7b_n16_mini64_bsz128_len8_resp8.log 2>&1 &

NNODES=2 WANDB_RUN_ID=run_20260121_6686c1c5 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes32_video_7b_n16_mini64_bsz128_len4_resp8.log 2>&1 &



NNODES=2 nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes8_offline 8 8 fsdp2 1.5 1 16 5e-5 0 > logs_run/ray2_scale_flash_forzen_filter_tie_acc_cost_enc0.2_cen0.5_nframes8_offline_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len8_resp8_lr5e-5.log 2>&1 &

NNODES=2 nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_tie_acc_cost_enc0.1_cen0.4_nframes8_offline 8 8 fsdp2 1.5 1 16 1e-4 0 > logs_run/ray2_scale_flash_forzen_filter_tie_acc_cost_enc0.1_cen0.4_nframes8_offline_fsdp2_7b_s16_n1_min0.2_max_1.5_mini32_bsz128_len8_resp8_lr1e-4.log 2>&1 &

# NNODES=2 WANDB_RUN_ID=run_20260108_6948bffb nohup bash visionthink/predictor/main_ray.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/ray2_scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8.log 2>&1 &

NNODES=2 WANDB_RUN_ID=run_20260110_51e327b8 nohup bash visionthink/predictor/main_ray.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/ray2_scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-6.log 2>&1 &

NNODES=2 WANDB_RUN_ID=run_20260110_b0df08e8 nohup bash visionthink/predictor/main_ray.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 1e-5 0 > logs_run/ray2_scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_1e-5.log 2>&1 &

NNODES=2 WANDB_RUN_ID=run_20260112_f5f7fab1 nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/ray2_scale_flash_ispred_filter_tie_acc_cost_enc0.15_cen0.5_nframes8_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

NNODES=8 nohup bash visionthink/predictor/main.sh 7B r1_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/r1_ray8_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini16_bsz64_len4_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B r1_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/r1_ray4_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini32_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B r1_sft_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/r1_sft_ray4_scale_flash_filter_tie_acc_cost_enc0.15_cen0.5_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini32_bsz128_len8_resp8_5e-6.log 2>&1 &


NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_ray4_pt_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len4_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_scale_gate_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_ray4_pt_scale_gate_filter_tie_acc_cost_enc0.2_cen0.5_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_hadw_asym_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_ray4_pt_hadw_asym_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_norm_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/mix_ray4_pt_asym_norm_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_m_norm_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-6 0 > logs_run/mix_ray4_pt_asym_m_norm_scale_gate_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_hadw_asym_h_scale_gate_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.3_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_hadw_asym_h_scale_gate_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.3_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_s_scale_gate_ispred_filter_frozen_newtie_acc_cost_enc0.1_cen0.3_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_asym_s_scale_gate_ispred_filter_frozen_newtie_acc_cost_enc0.1_cen0.3_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &






NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev1_sim_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_awa_scalev1_sim_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev1_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_awa_scalev1_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 WANDB_RUN_ID=run_20260129_ddb6c150 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev1b_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-5 0 > logs_run/mix_ray4_pt_asym_scalev1b_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev3b_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scalev3b_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awne_scalev3b_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_awne_scalev3b_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev3bd0_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scalev3bd0_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awne_scalev3bd0_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_awne_scalev3bd0_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awrk_swi_scalev3bd0_sim1_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_awrk_swi_scalev3bd0_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smol_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-4 0 > logs_run/mix_ray4_pt_asym_scale_smol_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-4.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smol_sim1_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-4 0 > logs_run/mix_ray4_pt_asym_scale_smol_simm_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-4.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv2_sim1_ent_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-4 0 > logs_run/mix_ray4_pt_asym_scale_smolv2_sim_ent_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-4.log 2>&1 &

NNODES=4 WANDB_RUN_ID=run_20260206_ca3698ba nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes128_video_mrope 4 8 fsdp2 2.0 1 16 2e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-5.log 2>&1 &

NNODES=4 WANDB_RUN_ID=run_20260211_a900135a nohup bash visionthink/predictor/main.sh 7B mix_sft_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_sft_asym_scale_smolv1_sim_filter_frozen_newtie_acc_cost_enc0.25_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope 7 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len7_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-5.log 2>&1 &

NNODES=4 WANDB_RUN_ID=run_20260211_a900135a nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 2e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_2e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awid_swi_scale_smolv1_sim_ccen_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_awid_swi_scale_smolv1_sim_ccen_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolhead_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolhead_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolembed_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolembed_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_5e-5.log 2>&1 &


### qwen3-vl-4b
NNODES=2 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes64_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes64_video_7b_n16_mini64_bsz128_len8_resp8.log 2>&1 &


NNODES=2 MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct nohup bash visionthink/predictor/main.sh 4B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes32_video_4b_n16_mini64_bsz128_len4_resp8.log 2>&1 &


NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv3_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-5.log 2>&1 &





NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev2_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_asym_scalev2_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev2_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_awa_scalev2_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_hadw_asym_scalev2_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_hadw_asym_scalev2_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev2b_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_asym_scalev2b_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev2bd_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-4 0 > logs_run/mix_ray4_pt_awa_scalev2bd_sim_filter_frozen_acc_cost_enc0.25_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-4.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_asym_scalev2bd_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 1e-4 0 > logs_run/mix_ray4_asym_scalev2bd_sim_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_1e-4.log 2>&1 &

# NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev2b_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 0 > logs_run/mix_ray4_pt_awa_scalev2b_sim_filter_frozen_acc_cost_enc0.15_cen0.35_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_2.0_mini64_bsz128_len4_resp8_2e-6.log 2>&1 &




NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_sft_scale_flash_filter_newtie_acc_cost_enc0.2_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_sft_ray4_scale_flash_filter_newtie_acc_cost_enc0.2_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini128_bsz256_len8_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_sft_hadw_scale_flash_ispred_filter_newtie_acc_cost_enc0.2_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_scale_flash_ispred_filter_newtie_acc_cost_enc0.2_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &


NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_sft_hadw_scale_flash_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_scale_flash_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_sft_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 bash visionthink/predictor/main.sh 7B mix_sft_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope 8 8 fsdp2 1.8 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes64_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &


NNODES=5 bash visionthink/predictor/main.sh 7B mix_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 debug >

NNODES=4 bash visionthink/predictor/main.sh 7B mix_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=4 bash visionthink/predictor/main.sh 7B mix_sft_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 0 > logs_run/mix_ray4_sft_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_5e-6.log 2>&1 &

NNODES=5 bash visionthink/predictor/main.sh 7B mix_sft_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 debug > a.log  2>&1


### DEBUG

NNODES=2 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 debug > d.log 2>&1

NNODES=2 nohup bash visionthink/predictor/main.sh 7B mix_base_nframes64_video 4 8 fsdp2 1.5 16 1 1e-5 debug > a.log 2>&1

NNODES=8 bash visionthink/predictor/main.sh 7B mix_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 debug > c.log 2>&1

NNODES=4 bash visionthink/predictor/main.sh 7B mix_hadw_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 debug > c.log 2>&1

NNODES=6 bash visionthink/predictor/main.sh 7B mix_hadw_sep_scale_gate_ispred_filter_newtie_acc_cost_enc0.2_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 5e-6 debug > c.log 2>&1 

NNODES=4 nohup bash visionthink/predictor/main.sh 7B mix_pt_awa_scalev2_sim_ispred_filter_frozen_newtie_acc_cost_enc0.15_cen0.35_nframes32_video_mrope 4 8 fsdp2 2.0 1 16 2e-6 debug > a.log 2>&1



# images

## base

nohup bash visionthink/predictor/run_3b.sh  > logs/pred_flash_3b_s4_n4_min0.2_max_3.0_mini32_bsz128_len2048.log 2>&1 &

nohup bash visionthink/predictor/run.sh  > logs/pred_3b_s4_n4_min0.2_max_2.5_mini32_bsz128_len2048.log 2>&1 &

nohup bash visionthink/predictor/run_7b.sh  > logs/pred_7b_s4_n4_min0.2_max_2.0_mini32_bsz128.log 2>&1 &

nohup bash visionthink/predictor/base.sh  > logs/base_3b_s16_mini32_bsz128.log 2>&1 &

nohup bash visionthink/predictor/base_7b.sh  > logs/base_7b_s16_mini32_bsz128.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base 8 2.5 16 0 2 0 0 > logs/base_7b_n16_mini32_bsz128_len8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B base 8 2.5 16 0 2 0 0 > logs/base_3b_n16_mini32_bsz128_len8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base 4 8 fsdp2 2.0 16 0 2 0 0 > logs/base_7b_n16_mini32_bsz128_len4_resp8.log 2>&1 &


nohup bash visionthink/predictor/main.sh 7B scale_flash 2 2.5 4 4 0 2 0 0 > logs/pred_flash_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len2048.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash 2 3.0 4 4 0 0 > logs/pred_flash_3b_s4_n4_min0.2_max_3.0_mini32_bsz128_len2048.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash 4 2.5 4 4 0 2 0 0 > logs/pred_flash_3b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4096.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash 4 2.5 4 4 1 2 0 0 > logs/pred_flash_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_filter.log 2>&1 &

# nohup bash visionthink/predictor/main.sh 3B scale_flash 2 3.0 8 1 0 2 0 0 > logs/pred_flash_3b_s1_n8_min0.2_max_3.0_mini32_bsz128_len2048.log 2>&1 &

nohup bash visionthink/predictor/base_7b.sh  > logs/base_7b_s16_mini32_bsz128_len8_verl.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash 8 fsdp 2.0 4 4 1 2 0 0 > logs/pred_flash_7b_s4_n4_min0.2_max_2.0_mini32_bsz128_len8_filter_fsdp.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base 8 fsdp 1.5 16 0 2 0 0 > logs/base_7b_n16_mini32_bsz128_len8_fsdp.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B base 4 8 fsdp2 1.5 16 0 2 0 0 > logs/base_7b_n16_mini32_bsz128_len4_resp8_fsdp2.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B base 4 8 fsdp2 2 16 0 2 0 0 > logs/base_3b_n16_mini32_bsz128_len4_resp8_fsdp2.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter 4 8 fsdp2 2.5 4 4 2 0 0 > logs/scale_flash_filter_fsdp2_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_cost0.25 4 8 fsdp2 2.5 4 4 2 0 0 > logs/scale_flash_filter_cost0.25_fsdp2_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash_filter 4 8 fsdp2 2.5 4 4 2 0 0 > logs/scale_flash_filter_fsdp2_3b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash_filter_cost0.25 4 8 fsdp2 2.5 4 4 2 0 0 > logs/scale_flash_filter_cost0.25_fsdp2_3b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_cost0.1_enc0.02 4 8 fsdp2 2.5 4 4 1e-5 0 > logs/scale_flash_filter_cost0.1_enc0.02_fsdp2_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash_filter_cost0.05_enc0.01 4 8 fsdp2 2.5 4 4 1e-5 2 0 0 > logs/scale_flash_filter_cost0.05_enc0.01_fsdp2_3b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_cost0.05_enc0.02 4 8 fsdp2 2.5 4 4 1e-4 2 0 0 > logs/scale_flash_frozen_filter_cost0.05_enc0.02_fsdp2_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8_lr1e-4.log 2>&1 &


scp -rP 26173 root@210.75.240.147:/mnt/userdata/llm/plt .
## update multi-image and scales

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter 4 8 fsdp2 2.5 4 4 1e-5 2 0 0 > logs1/scale_flash_filter_fsdp2_7b_s4_n4_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_flash_filter_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter 4 8 fsdp2 2.0 2 8 1e-5 2 0 0 > logs/scale_flash_filter_fsdp2_7b_s8_n2_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_cost0.1_enc0.02 4 8 fsdp2 2.0 2 8 1e-5 0 > logs/scale_flash_filter_cost0.1_enc0.02_fsdp2_7b_s8_n2_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_cost0.1_enc0.02 4 8 fsdp2 2.0 2 8 1e-4 2 0 0 > logs/scale_flash_frozen_filter_cost0.1_enc0.02_fsdp2_7b_s8_n2_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr1e-4.log 2>&1 &

nohup bash visionthink/predictor/main.sh 3B scale_flash_filter_cost0.05_enc0.02 4 8 fsdp2 2.5 1 16 1e-5 2 0 0 > logs/scale_flash_filter_cost0.05_enc0.02_fsdp2_3b_s16_n1_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_mul_cost_enc0.25 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_flash_filter_mul_cost_enc0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_abs_cost0.1 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_flash_filter_abs_cost0.1_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_mul_cost_enc0.02 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_flash_filter_mul_cost_enc0.02_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_mul_cost_enc0.02 4 8 fsdp2 2.0 2 8 1e-5 2 0 0 > logs/scale_flash_filter_mul_cost_enc0.02_fsdp2_7b_s8_n2_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter 4 8 fsdp2 2.0 1 16 1e-4 2 0 0 > logs/scale_flash_frozen_filter_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr1e-4.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_mul_cost_enc0.01 4 8 fsdp2 2.0 1 16 5e-5 2 0 0 > logs/scale_flash_frozen_filter_mul_cost_enc0.01_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr5e-5.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25 4 8 fsdp2 2.0 1 16 1e-5 0 > logs/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_tie_acc_cost_enc0.25_cen0.5 4 8 fsdp2 2.0 1 16 1e-4 0 > logs/scale_flash_frozen_filter_tie_acc_cost_enc0.25_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr1e-4.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.25 4 8 fsdp2 2.0 1 16 1e-5 0 > logs/scale_flash_filter_tie_acc_cost_enc0.1_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_frozen_filter_tie_acc_cost_enc0.25_cen0.25 4 8 fsdp2 2.0 1 16 5e-5 0 > logs/scale_flash_frozen_filter_tie_acc_cost_enc0.25_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr5e-5.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.25 4 8 fsdp2 2.0 1 16 5e-6 0 > logs/scale_flash_filter_tie_acc_cost_enc0.2_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.2_cen0.5 4 8 fsdp2 2.0 1 16 5e-6 0 > logs/scale_flash_filter_tie_acc_cost_enc0.2_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.25 4 8 fsdp2 2.0 1 16 5e-6 0 > logs/scale_flash_filter_tie_acc_cost_enc0.1_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_tie_acc_cost_enc0.1_cen0.5 4 8 fsdp2 2.0 1 16 1e-5 0 > logs/scale_flash_filter_tie_acc_cost_enc0.1_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_1e-5_vllm.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.25_cen0.5 4 8 fsdp2 2.0 1 16 2e-6 0 > logs/scale_flash_ispred_filter_tie_acc_cost_enc0.25_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_2e-6_vllm.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.2_cen0.5 4 8 fsdp2 2.0 1 16 5e-6 0 > logs/scale_flash_ispred_filter_tie_acc_cost_enc0.2_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_flash_ispred_filter_tie_acc_cost_enc0.5_cen0.5 4 8 fsdp2 2.0 1 16 1e-6 0 > logs/scale_flash_ispred_filter_tie_acc_cost_enc0.5_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_1e-6.log 2>&1 &

## discrete

nohup bash visionthink/predictor/main.sh 7B scale_disc_filter_mul_cost_enc0.05 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_disc_filter_mul_cost_enc0.05_fsdp2_7b_s16_n1_min0.25_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_disc_filter 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_disc_filter_fsdp2_7b_s16_n1_min0.25_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_disc_filter_tie_cost_enc0.05 4 8 fsdp2 2.0 1 16 1e-5 2 0 0 > logs/scale_disc_filter_tie_cost_enc0.05_fsdp2_7b_s16_n1_min0.25_max_2.0_mini32_bsz128_len4_resp8.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_disc_filter_tie_acc_cost_enc0.25_cen0.25 4 8 fsdp2 2.0 1 16 5e-5 0 > logs/scale_disc_filter_tie_acc_cost_enc0.25_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr5e-5.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_disc_filter_tie_acc_cost_enc0.2_cen0.25 4 8 fsdp2 2.0 1 16 5e-6 0 > logs/scale_disc_filter_tie_acc_cost_enc0.25_cen0.25_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_lr5e-6.log 2>&1 &

nohup bash visionthink/predictor/main.sh 7B scale_disc_ispred_filter_tie_acc_cost_enc0.15_cen0.5 4 8 fsdp2 2.0 1 16 1e-6 0 > logs/scale_disc_ispred_filter_tie_acc_cost_enc0.15_cen0.5_fsdp2_7b_s16_n1_min0.2_max_2.0_mini32_bsz128_len4_resp8_1e-6.log 2>&1 &

## ray

NNODES=2 nohup bash visionthink/predictor/main.sh 7B scale_flash_filter_cost0.05_enc0.02 4 8 fsdp2 2.0 2 8 1e-5 0 > logs/ray2_scale_flash_filter_cost0.05_enc0.02_fsdp2_7b_s8_n2_min0.2_max_2.0_mini64_bsz256_len4_resp8.log 2>&1 &

NNODES=2 nohup bash visionthink/predictor/main.sh 3B scale_flash_filter 4 8 fsdp2 2.5 2 8 1e-5 2 0 0 > logs/ray2_scale_flash_filter_fsdp2_3b_s8_n2_min0.2_max_2.5_mini32_bsz128_len4_resp8.log 2>&1 &
