#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅提取 + 合并日志数据（无绘图）
- 内嵌日志解析逻辑（无需单独 log_extract.py）
- 自动处理多个实验文件夹
- 结果按算法分组，带标注行和空行分隔
- ✅ 新增字段：max_memory_allocated (GB)
"""

import os
import re
import pandas as pd
from pathlib import Path


# ================================
# 日志解析逻辑（增强版：含 max_memory_allocated）
# ================================

def extract_info_from_log(file_path):
    peak_memory_mb = None
    max_reserved_gb = None
    max_allocate_gb = None  # ✅ 新增
    avg_time = None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path.name}: {e}")
        return (file_path.name, None, None, None, None)  # ✅ 多一个返回值

    # 1. 显存峰值 (MB)
    for line in lines:
        if "模型第一次反向传播阶段的显存峰值（所有模块中的最大值）:" in line:
            match = re.search(r':\s*([\d.]+)\s*MB', line)
            if match:
                peak_memory_mb = float(match.group(1))
            break

    # 2. max_memory_reserved (GB) —— 取最后一个
    for line in reversed(lines):
        if "torch.cuda.max_memory_reserved()" in line:
            match = re.search(r'torch\.cuda\.max_memory_reserved\(\)\s*:\s*([\d.]+)', line)
            if match:
                max_reserved_gb = float(match.group(1))
            break

    # ✅ 3. 新增：max_memory_allocated (GB) —— 取最后一个
    for line in reversed(lines):
        if "torch.cuda.max_memory_allocated()" in line:
            match = re.search(r'torch\.cuda\.max_memory_allocated\(\)\s*:\s*([\d.]+)', line)
            if match:
                max_allocate_gb = float(match.group(1))
            break

    # 4. 平均时间 (s)
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
        max_allocate_gb,  # ✅
        avg_time
    )


def natural_sort_key(filename):
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


def process_folder_logs(folder_path: Path):
    """在指定文件夹内提取所有 .log 数据，返回 DataFrame（含 max_memory_allocated）"""
    log_files = list(folder_path.glob("*.log"))
    if not log_files:
        print(f"⚠️ {folder_path.name}: 无 .log 文件")
        return pd.DataFrame()

    results = []
    for log_file in log_files:
        info = extract_info_from_log(log_file)
        results.append(info)
        print(f"✅ {folder_path.name} ← {info[0]}")

    if not results:
        return pd.DataFrame()

    # ✅ 更新列名：新增 'max_memory_allocated (GB)'
    df = pd.DataFrame(
        results,
        columns=[
            "文件名",
            "反向传播显存峰值 (MB)",
            "max_memory_reserved (GB)",
            "max_memory_allocated (GB)",  # ✅
            "策略后平均 epoch 时间 (s)"
        ]
    )
    df = df.sort_values(by="文件名", key=lambda col: col.map(natural_sort_key)).reset_index(drop=True)
    return df


# ================================
# 主合并逻辑（兼容新增字段）
# ================================

def merge_experiments(experiment_folders, output_file="merged_results.csv"):
    all_data = []

    for folder in experiment_folders:
        folder_path = Path(folder).resolve()
        if not folder_path.is_dir():
            print(f"⚠️ 跳过不存在的文件夹: {folder}")
            continue

        print(f"\n📁 处理实验: {folder_path.name}")
        df = process_folder_logs(folder_path)

        if df.empty:
            print(f"⚠️ {folder_path.name}: 未提取到有效数据")
            continue

        # 插入算法标注行（与 df 列数对齐）
        algo_row = pd.DataFrame([{df.columns[0]: f"Algorithm: {folder_path.name}"}])
        for col in df.columns[1:]:
            algo_row[col] = pd.NA
        combined = pd.concat([algo_row, df], ignore_index=True)
        all_data.append(combined)

        # 添加空行分隔（列数需匹配 df）
        empty_row = pd.DataFrame([[pd.NA] * len(df.columns)], columns=df.columns)
        all_data.append(empty_row)

    if not all_data:
        print("❌ 无任何数据可合并。")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 合并完成！共处理 {len(experiment_folders)} 个实验，结果已保存至:\n{output_file}")


# ================================
# 入口
# ================================

if __name__ == "__main__":
    experiment_folders = [
        "./useful_old2/cricket_logs_checkmate",
        "./useful_old2/cricket_logs_oursILP",
        "./useful_old2/cricket_logs_monet",
        "./useful_old2/cricket_logs_mimose",
    ]

    print("🚀 开始日志提取与合并...")
    merge_experiments(
        experiment_folders=experiment_folders,
        output_file="merged_results.csv"
    )