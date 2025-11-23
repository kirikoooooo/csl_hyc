import os
import re
import pandas as pd
from pathlib import Path

def extract_info_from_log(file_path):
    """
    从单个日志文件中提取三项关键信息。
    返回: (filename, peak_memory_mb, max_reserved_gb, avg_time)
    """
    peak_memory_mb = None
    max_reserved_gb = None
    avg_time = None
    max_allocate_gb = None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 1. 提取 "模型第一次反向传播阶段的显存峰值（所有模块中的最大值）"
    for line in lines:
        if "模型第一次反向传播阶段的显存峰值（所有模块中的最大值）:" in line:
            match = re.search(r':\s*([\d.]+)\s*MB', line)
            if match:
                peak_memory_mb = float(match.group(1))
            break

    # 2. 提取最后一个 torch.cuda.max_memory_reserved()
    for line in reversed(lines):
        if "torch.cuda.max_memory_reserved()" in line:
            match = re.search(r'torch\.cuda\.max_memory_reserved\(\)\s*:\s*([\d.]+)', line)
            if match:
                max_reserved_gb = float(match.group(1))
            break
    for line in reversed(lines):
        if "torch.cuda.max_memory_allocated()" in line:
            match = re.search(r'torch\.cuda\.max_memory_allocated\(\)\s*:\s*([\d.]+)', line)
            if match:
                max_allocate_gb = float(match.group(1))
            break

    # 3. 提取平均时间
    for line in lines:
        if "!!!采用策略后的几个epoch 平均时间:" in line:
            match = re.search(r'!!!采用策略后的几个epoch 平均时间:\s*([\d.]+)', line)
            if match:
                avg_time = float(match.group(1))
            break

    return (
        os.path.basename(file_path),
        peak_memory_mb,
        max_reserved_gb,
        max_allocate_gb,
        avg_time
    )


def natural_sort_key(filename):
    """自然排序 key，支持 2GB < 2.5GB < 10GB"""
    s = filename.lower()
    parts = re.split(r'(\d+\.?\d*)', s)
    result = []
    for part in parts:
        if part.replace('.', '', 1).isdigit():
            try:
                result.append(float(part))
            except ValueError:
                result.append(part)
        else:
            result.append(part)
    return result


def main():
    current_dir = Path(__file__).parent.resolve()  # 脚本所在目录
    log_files = list(current_dir.glob("*.log"))

    if not log_files:
        print("⚠️ 当前目录下没有 .log 文件！")
        return

    results = []
    for log_file in log_files:
        try:
            info = extract_info_from_log(log_file)
            results.append(info)
            print(f"✅ 已处理: {info[0]}")
        except Exception as e:
            print(f"❌ 处理失败 {log_file.name}: {e}")

    if not results:
        print("❌ 未成功提取任何日志数据。")
        return

    # 创建 DataFrame 并自然排序
    df = pd.DataFrame(
        results,
        columns=["文件名", "反向传播显存峰值 (MB)", "max_memory_reserved (GB)","max_memory_allocated", "策略后平均 epoch 时间 (s)"]
    )
    df = df.sort_values(by="文件名", key=lambda col: col.map(natural_sort_key)).reset_index(drop=True)

    # 输出到当前目录
    output_path = current_dir / "log_summary.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 共处理 {len(results)} 个日志文件，结果已保存至:\n{output_path}")


if __name__ == "__main__":
    main()