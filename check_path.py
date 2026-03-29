import os
import datasets
from tqdm import tqdm
from qwen_vl_utils import fetch_video
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# 过滤警告
warnings.filterwarnings("ignore")

def validate_single_sample(args):
    """
    单个样本的处理函数，供线程池调用
    """
    idx, row = args
    errors = []
    
    # 获取 doc_id
    doc_id = row.get("extra_info", {}).get("doc_id", "Unknown")
    
    # 获取路径列表
    video_paths = row.get("videos", [])
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    elif video_paths is None:
        video_paths = []
        
    for video_path in video_paths:
        # 1. 基础路径检查
        if not os.path.exists(video_path["video"]):
            errors.append({
                "index": idx,
                "doc_id": doc_id,
                "path": video_path,
                "error": "File not found (os.path.exists failed)"
            })
            continue

        # 2. 深度加载检查
        try:
            # 尝试加载视频
            _ = fetch_video(video_path)
        except Exception as e:
            errors.append({
                "index": idx,
                "doc_id": doc_id,
                "path": video_path,
                "error": str(e)
            })
    
    return errors

def check_video_integrity_multithread(parquet_path, max_workers=16):
    print(f"正在加载数据集: {parquet_path}")
    try:
        dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print(f"数据集加载成功，共 {len(dataset)} 条数据。")
    print(f"开始多线程校验 (Threads={max_workers})...")

    all_error_files = []
    
    # 准备任务参数
    tasks = []
    for idx, row in enumerate(dataset):
        tasks.append((idx, row))

    # 启动线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(validate_single_sample, task): task[0] for task in tasks}
        
        # 使用 tqdm 监控完成的任务
        progress_bar = tqdm(as_completed(future_to_idx), total=len(tasks), desc="Checking Videos")
        
        for future in progress_bar:
            try:
                # 获取该样本的检测结果（返回的是错误列表，为空说明正常）
                sample_errors = future.result()
                
                if sample_errors:
                    all_error_files.extend(sample_errors)
                    
                    # 实时打印错误信息 (使用 tqdm.write 防止破坏进度条)
                    for err in sample_errors:
                        tqdm.write(f"\n[Error] Idx: {err['index']} | ID: {err['doc_id']}")
                        tqdm.write(f"  Path: {err['path']}")
                        tqdm.write(f"  Msg:  {err['error']}\n")
                        
            except Exception as e:
                tqdm.write(f"Task exception: {e}")

    # --- 输出最终报告 ---
    print("\n" + "="*50)
    print("校验完成报告")
    print("="*50)
    
    if len(all_error_files) == 0:
        print("\n✅ 所有视频文件均校验通过！")
    else:
        print(f"\n❌ 共发现 {len(all_error_files)} 个错误的视频文件。")
        print("-" * 50)
        
        # 保存详细日志
        log_file = "corrupt_videos_log.txt"
        with open(log_file, "w") as f:
            f.write("Index\tDoc_ID\tPath\tError_Message\n")
            for item in all_error_files:
                # 移除换行符防止日志格式混乱
                clean_msg = item['error'].replace('\n', ' ').replace('\r', '')
                f.write(f"{item['index']}\t{item['doc_id']}\t{item['path']}\t{clean_msg}\n")
        
        print(f"完整错误列表已保存至: {log_file}")

if __name__ == "__main__":
    PARQUET_FILE = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/train.parquet"
    
    # 建议设置线程数为 8-32 之间，取决于你的磁盘 I/O 能力
    # 如果是 SSD 可以设置大一点 (16-32)，如果是机械硬盘或者网络挂载盘，设置小一点 (4-8)
    WORKERS = 32
    
    if os.path.exists(PARQUET_FILE):
        check_video_integrity_multithread(PARQUET_FILE, max_workers=WORKERS)
    else:
        print(f"错误: Parquet 文件不存在 -> {PARQUET_FILE}")