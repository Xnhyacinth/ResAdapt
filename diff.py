import torch
from transformers import AutoModel, AutoConfig
import os

def load_model_or_state_dict(path):
    """
    尝试加载模型。优先尝试 HuggingFace AutoModel，
    如果失败（例如只是一个 .pt 文件），则尝试直接加载 state_dict。
    """
    print(f"正在加载: {path} ...")
    try:
        # 尝试作为 HF 模型加载 (不加载权重以节省内存，主要为了获取结构，或者使用 device_map)
        # 注意：为了对比权重，我们需要加载实际参数
        model = AutoModel.from_pretrained(path, trust_remote_code=True, device_map="cpu")
        return model.state_dict()
    except Exception as e_hf:
        print(f"HF加载模式跳过，尝试直接加载 state_dict... ({e_hf})")
        try:
            # 针对可能是 deepspeed 保存的 mp_rank_00_model_states.pt 或 pytorch_model.bin
            # 这里假设如果是文件夹，可能需要指定具体文件名，或者它是一个标准的 .pt 文件
            if os.path.isdir(path):
                # 常见文件名猜测
                candidates = ["pytorch_model.bin", "mp_rank_00_model_states.pt", "model.safetensors"]
                file_path = None
                for c in candidates:
                    p = os.path.join(path, c)
                    if os.path.exists(p):
                        file_path = p
                        break
                if file_path is None:
                    raise FileNotFoundError(f"在目录中未找到常见的权重文件: {path}")
            else:
                file_path = path
            
            state_dict = torch.load(file_path, map_location="cpu")
            # 处理 DeepSpeed/Megatron 可能包裹的 'module' key
            if "module" in state_dict:
                state_dict = state_dict["module"]
            return state_dict
        except Exception as e_direct:
            print(f"加载失败: {e_direct}")
            return None

def check_weight_change(ckpt_sd, orig_sd, target_keywords):
    print("\n" + "="*50)
    print(f"开始对比权重，关注关键词: {target_keywords}")
    print("="*50)

    changed_modules = set()
    unchanged_modules = set()
    missing_modules = set()

    # 遍历 Checkpoint 中的所有 key
    for key in ckpt_sd.keys():
        # 检查是否包含我们关注的关键词 (predictor, vision_tower 等)
        is_target = any(k in key for k in target_keywords)
        
        if not is_target:
            continue

        if key not in orig_sd:
            missing_modules.add(key)
            print(f"[NEW] {key} (在原始模型中不存在)")
            continue

        # 获取权重张量
        w_ckpt = ckpt_sd[key]
        w_orig = orig_sd[key]

        # 确保形状一致
        if w_ckpt.shape != w_orig.shape:
            print(f"[SHAPE MISMATCH] {key}: Checkpoint {w_ckpt.shape} vs Original {w_orig.shape}")
            continue

        # 对比数值 (使用 allclose 或直接相减)
        # 考虑到浮点误差，这里判断差值是否全为 0
        if torch.equal(w_ckpt, w_orig):
            unchanged_modules.add(key)
        else:
            diff = (w_ckpt - w_orig).abs().max().item()
            if diff > 0:
                changed_modules.add(key)
                # 打印变化，只打印一部分避免刷屏
                print(f"[CHANGED] {key} | Max Diff: {diff:.6f}")
            else:
                unchanged_modules.add(key)

    print("\n" + "="*50)
    print("总结报告")
    print("="*50)
    
    # 简单的聚合统计
    keywords_stats = {k: {"changed": 0, "unchanged": 0} for k in target_keywords}
    
    for k in target_keywords:
        print(f"\n检查模块关键字: '{k}'")
        
        # 统计该关键字下的变化
        k_changed = [x for x in changed_modules if k in x]
        k_unchanged = [x for x in unchanged_modules if k in x]
        
        if k_changed:
            print(f"  -> 🔴 发现权重变化 ({len(k_changed)} 层):")
            print(f"     示例: {k_changed[:2]} ...")
        else:
            print("  -> 🟢 未发现权重变化 (Weights Frozen/Unchanged)")
            
        if k_unchanged:
            print(f"  -> ⚪ 保持不变 ({len(k_unchanged)} 层)")

# ================= 配置路径 =================
# 1. Checkpoint 路径 (您提供的)
ckpt_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/debug/Gen-3B-ray-bsz128-mini32-n8-min0.25-max1.25-scale/global_step_10"
# ckpt_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/new_qwen2.5_vl-7b" 
# 2. 原始模型路径 (请在此处填入 Base Model 的路径)
# 例如: "/mnt/bn/.../models/Qwen-VL-Chat" 或 "Qwen/Qwen-VL"
# original_model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor" 
original_model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/my_qwen2.5_vl-3b" 

# ================= 执行对比 =================
if original_model_path == "YOUR_ORIGINAL_MODEL_PATH_HERE":
    print("请先在脚本中设置 'original_model_path' 变量！")
else:
    ckpt_sd = load_model_or_state_dict(ckpt_path)
    orig_sd = load_model_or_state_dict(original_model_path)

    if ckpt_sd and orig_sd:
        # 关注的组件关键词
        targets = ["predictor", "vision_tower", "language_model"]
        check_weight_change(ckpt_sd, orig_sd, targets)
    breakpoint()

