
bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 290 290 all 1 all


bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf128_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_2e-5 460 470 all 1 all



bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf128_vidm-dp2-s16-7B-bsz128-mini64-n1-max1.8-l8-r8_2e-5 460 470 all 1 base_mrope_qwen3vl_8B

bash eval.sh /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 290 290 all 0 base_mrope_qwen3vl_8B




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


