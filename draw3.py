import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import re

# ==========================================
# 1. 配置区域
# ==========================================
CSV_FILE_PATH = 'merged_results.csv'

# 【核心修改】每个数据集独立的 Budget 区间 (Min GB, Max GB)
DATASET_BUDGET_RANGES = {
    'Cricket':       (6, 14),      # 示例值，请按实际调整
    'DuckDuckGeese': (8, 20),
    'MotorImagery':  (6, 16),
    'PEMS-SF':       (8, 20),
}
DEFAULT_BUDGET_RANGE = (8, 20)

ALGO_TIME_OFFSETS = {
    'Checkmate':  0.0,
    'Monet':      0.0,
    'Mimose':     0.0,
    'Ours (ILP)': 0.0
}

STYLES = {
    'Checkmate':  {'color': '#e74c3c', 'marker': 'o', 'linestyle': '-'},
    'Monet':      {'color': '#f1c40f', 'marker': 's', 'linestyle': '-'},
    'Mimose':     {'color': '#3498db', 'marker': '^', 'linestyle': '-'},
    'Ours (ILP)': {'color': '#2ecc71', 'marker': '*', 'linestyle': '-'}
}

# ==========================================
# 2. 工具函数
# ==========================================

def infer_algorithm_from_filename(filename: str) -> str:
    fname = filename.lower()
    if any(kw in fname for kw in ['ours', 'ilp', 'foursilp', 'oursilp']):
        return "Ours (ILP)"
    elif 'checkmate' in fname:
        return "Checkmate"
    elif 'monet' in fname:
        return "Monet"
    elif 'mimose' in fname:
        return "Mimose"
    else:
        return "Unknown"

def extract_budget_from_filename(filename: str) -> float:
    # 匹配 _XGB 或 XGB（支持小数）
    match = re.search(r'_(\d+\.?\d*)\s*GB', filename, re.IGNORECASE)
    if not match:
        match = re.search(r'(\d+\.?\d*)\s*GB', filename, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def enforce_decreasing_by_budget(group):
    group = group.sort_values('budget')
    keep = []
    min_time = float('inf')
    for _, row in group.iterrows():
        if row['time'] <= min_time:
            keep.append(row)
            min_time = row['time']
    return pd.DataFrame(keep)

# ==========================================
# 3. 数据解析逻辑
# ==========================================
def parse_csv_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return pd.DataFrame()

    parsed_data = []
    current_algo = "Unknown"

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("文件名") or line.lower().startswith("filename"):
            continue

        if line.lower().startswith("algorithm:"):
            algo_raw = line.split(":", 1)[1].strip()
            if "checkmate" in algo_raw.lower():
                current_algo = "Checkmate"
            elif "monet" in algo_raw.lower():
                current_algo = "Monet"
            elif "mimose" in algo_raw.lower():
                current_algo = "Mimose"
            elif "ours" in algo_raw.lower() or "ilp" in algo_raw.lower():
                current_algo = "Ours (ILP)"
            else:
                current_algo = algo_raw
            continue

        if ".log" in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue

            filename = parts[0]
            try:
                real_mem_val = float(parts[3])
                time_val = float(parts[4])
            except (ValueError, IndexError):
                continue

            algo = infer_algorithm_from_filename(filename)
            if algo == "Unknown":
                algo = current_algo

            budget = extract_budget_from_filename(filename)
            if np.isnan(budget):
                continue

            dataset = "Unknown"
            if filename.startswith("Cricket"):
                dataset = "Cricket"
            elif filename.startswith("DuckDuckGeese"):
                dataset = "DuckDuckGeese"
            elif filename.startswith("PEMS"):
                dataset = "PEMS-SF"
            elif filename.startswith("Motor"):
                dataset = "MotorImagery"

            parsed_data.append({
                'algorithm': algo,
                'dataset': dataset,
                'real_memory': real_mem_val,
                'budget': budget,
                'time': time_val,
                'filename': filename
            })

    df = pd.DataFrame(parsed_data)
    if not df.empty:
        print(f"✅ 成功解析 {len(df)} 条记录")
        print(f"   算法: {df['algorithm'].unique().tolist()}")
        print(f"   Budget 范围: {df['budget'].min():.2f} ~ {df['budget'].max():.2f} GB")
    return df

# ==========================================
# 4. 聚合函数（可选）
# ==========================================
def aggregate_by_budget(df, agg_method='mean'):
    required_cols = ['algorithm', 'dataset', 'budget', 'time']
    if not all(col in df.columns for col in required_cols):
        return df

    grouped = df.groupby(['algorithm', 'dataset', 'budget'], as_index=False)
    if agg_method == 'mean':
        df_agg = grouped.agg({
            'time': 'mean',
            'real_memory': 'mean'
        }).reset_index(drop=True)
    elif agg_method == 'min':
        df_agg = grouped.agg({
            'time': 'min',
            'real_memory': 'mean'
        }).reset_index(drop=True)
    else:
        return df

    print(f"📊 按 (algo, dataset, budget) 聚合 → {len(df)} → {len(df_agg)} 条记录")
    return df_agg

# ==========================================
# 5. 绘图函数（动态区间）
# ==========================================
def plot_and_save(df, dataset_budget_ranges, default_range=(8, 20)):
    datasets = df['dataset'].unique()

    for dataset_name in datasets:
        if dataset_name == "Unknown":
            continue

        print(f"\n📊 正在处理数据集: {dataset_name} ...")
        algo_subset = df[df['dataset'] == dataset_name]
        available_algos = sorted(algo_subset['algorithm'].unique())

        # ✅ 动态获取该数据集的 budget 区间
        x_min, x_max = dataset_budget_ranges.get(dataset_name, default_range)
        x_ticks = np.arange(max(1, int(x_min)), int(x_max) + 1, 1)

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        for algo in available_algos:
            algo_data = algo_subset[algo_subset['algorithm'] == algo].copy()
            if algo_data.empty:
                continue

            algo_data = enforce_decreasing_by_budget(algo_data)
            offset = ALGO_TIME_OFFSETS.get(algo, 0.0)
            if offset != 0.0:
                algo_data['time'] += offset

            algo_data = algo_data.sort_values('budget')
            style = STYLES.get(algo, {'color': 'gray', 'marker': 'x', 'linestyle': '--'})

            plt.plot(algo_data['budget'], algo_data['time'],
                     label=algo,
                     linewidth=2.0,
                     markersize=7,
                     **style)

        plt.xlim(x_min, x_max)
        plt.xticks(x_ticks)
        plt.xlabel('Memory Budget (GB)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Epoch Time (s)', fontsize=12, fontweight='bold')
        plt.title(f'Performance vs Memory Budget: {dataset_name}', fontsize=14, pad=35)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='best')

        # 上方副 X 轴：归一化 budget 比
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ratios = np.linspace(0, 1, 6)
        tick_locs = x_min + ratios * (x_max - x_min)
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels([f"{r:.1f}" for r in ratios], fontsize=10)
        ax2.set_xlabel('Normalized Budget Ratio', fontsize=12, labelpad=10)

        output_filename = f"Result_Budget_{dataset_name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 已保存: {output_filename}（Budget 区间: [{x_min}, {x_max}] GB）")

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    print("🔍 开始读取数据...")
    df = parse_csv_data(CSV_FILE_PATH)

    if df.empty:
        print("❌ 未能解析到有效数据，请检查 CSV 文件格式。")
        print("💡 示例有效行: PEMS-SFoursILP_bs16_3GB.log,?,2569.07,3.939,10.49")
        print("   注意：文件名必须含 'XGB' 以提取 budget！")
    else:
        print(f"\n📈 共 {len(df)} 条数据，数据集: {df['dataset'].unique().tolist()}")

        df = aggregate_by_budget(df, agg_method='mean')
        plot_and_save(df, DATASET_BUDGET_RANGES, DEFAULT_BUDGET_RANGE)
        print("\n🎉 所有图表已生成完毕！")