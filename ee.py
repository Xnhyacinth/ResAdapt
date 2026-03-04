# import re
# import sys

# def extract_ttft_numbers(file_path):
#     """
#     从指定文件中提取 "ttft" 后面的数字（支持整数/浮点数，不区分大小写）
#     :param file_path: 目标文件路径
#     :return: 有效数字列表（空列表表示未找到）
#     """
#     ttft_pattern = re.compile(r'ttft\D*(\d+\.?\d*)', re.IGNORECASE)  # 正则匹配规则
#     numbers = []
    
#     try:
#         # 以 UTF-8 编码读取文件（避免中文/特殊字符报错）
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(f, 1):  # 遍历每一行，记录行号
#                 # 查找当前行中所有符合规则的数字
#                 matches = ttft_pattern.findall(line)
#                 for match in matches:
#                     try:
#                         # 将匹配到的字符串转为浮点数（兼容整数）
#                         num = float(match)
#                         numbers.append(num)
#                         # 可选：打印提取详情（方便调试）
#                         # print(f"行{line_num} 提取到 TTFT 数值: {num}")
#                     except ValueError:
#                         # 理论上正则已过滤非数字，此处为兜底
#                         print(f"警告：行{line_num} 匹配到无效数字 '{match}'，已跳过")
#     except FileNotFoundError:
#         print(f"错误：文件 '{file_path}' 不存在，请检查路径是否正确！")
#         sys.exit(1)
#     except PermissionError:
#         print(f"错误：无权限读取文件 '{file_path}'！")
#         sys.exit(1)
#     except Exception as e:
#         print(f"读取文件时发生未知错误：{str(e)}")
#         sys.exit(1)
    
#     return numbers

# def calculate_average(numbers):
#     """计算数字列表的平均数"""
#     if not numbers:
#         return None  # 无有效数字时返回 None
#     total = sum(numbers)
#     average = total / len(numbers)
#     return average

# def main():
#     # 1. 检查命令行参数（确保用户传入了文件路径）
#     if len(sys.argv) != 2:
#         print("用法：python calculate_ttft_average.py <目标文件路径>")
#         print("示例：python calculate_ttft_average.py ./experiment_log.txt")
#         sys.exit(1)
    
#     # 2. 提取 TTFT 数字
#     file_path = sys.argv[1]
#     print(f"正在从文件 '{file_path}' 中提取 TTFT 数值...")
#     ttft_numbers = extract_ttft_numbers(file_path)
    
#     # 3. 计算并输出结果
#     if not ttft_numbers:
#         print("未找到任何有效的 TTFT 数值！")
#         sys.exit(0)
    
#     average = calculate_average(ttft_numbers)
#     print(f"\n提取结果汇总：")
#     print(f"- 有效 TTFT 数值个数：{len(ttft_numbers)}")
#     print(f"- TTFT 数值列表：{ttft_numbers}")
#     print(f"- TTFT 平均数：{average:.2f}")  # 保留 2 位小数，可按需调整

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

ROOT = Path("/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify/"
            "r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5")

SRC = Path("/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/visionthink/predictor/modeling_predictor_smolv3.py")

STEP_RE = re.compile(r"^global_step_(\d+)$")

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")
    if not SRC.exists():
        raise FileNotFoundError(f"SRC not found: {SRC}")

    replaced = 0
    skipped_no_hf = 0
    skipped_no_dst = 0
    skipped_low_step = 0

    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue

        m = STEP_RE.match(child.name)
        if not m:
            continue

        step = int(m.group(1))
        if step < 300:
            skipped_low_step += 1
            continue

        hf_dir = child / "predictor" / "huggingface"
        if not hf_dir.is_dir():
            skipped_no_hf += 1
            continue

        dst = hf_dir / "modeling_predictor_smolv3.py"
        if not dst.exists():
            # 也可能你希望“即使没有也创建”，如需可改成允许创建
            skipped_no_dst += 1
            print(f"[SKIP] {child.name}: dst not found: {dst}")
            continue

        # 备份
        bak = dst.with_suffix(dst.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(dst, bak)

        # 覆盖
        shutil.copy2(SRC, dst)
        replaced += 1
        print(f"[OK] {child.name}: replaced {dst} (backup: {bak.name})")

    print("\n==== Summary ====")
    print(f"Replaced: {replaced}")
    print(f"Skipped (step<300): {skipped_low_step}")
    print(f"Skipped (no predictor/huggingface): {skipped_no_hf}")
    print(f"Skipped (dst missing): {skipped_no_dst}")

if __name__ == "__main__":
    main()