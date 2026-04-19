
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
CSV_FILE_PATH = 'merged_results.csv'

# 不同数据集的显存范围 (Min GB, Max GB)
DATASET_MEM_RANGES = {
    'Cricket':       (6, 14),
    'DuckDuckGeese': (8, 20),
    'MotorImagery':  (6, 16),
    'PEMS-SF':       (8, 20),
}
DEFAULT_MEM_RANGE = (8, 20)

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

def enforce_decreasing(df):
    if df.empty:
        return df
    df_sorted = df.sort_values('real_memory')
    keep_indices = []
    min_time_so_far = float('inf')
    for idx, row in df_sorted.iterrows():
        current_time = row['time']
        if current_time <= min_time_so_far:
            keep_indices.append(idx)
            min_time_so_far = current_time
    return df.loc[keep_indices].copy()

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
        if not line:
            continue

        if line.startswith("文件名") or line.lower().startswith("filename"):
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
                'time': time_val
            })

    df = pd.DataFrame(parsed_data)
    if not df.empty:
        print(f"✅ 成功解析 {len(df)} 条记录，含算法: {df['algorithm'].unique().tolist()}")
    return df

# ==========================================
# 4. 绘图函数
# ==========================================
def plot_and_save(df, dataset_mem_ranges, default_range=(8, 20)):
    datasets = df['dataset'].unique()

    for dataset_name in datasets:
        if dataset_name == "Unknown":
            continue

        algo_subset = df[df['dataset'] == dataset_name]
        available_algos = sorted(algo_subset['algorithm'].unique())
        mem_range = dataset_mem_ranges.get(dataset_name, default_range)
        x_min, x_max = mem_range

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        for algo in available_algos:
            algo_data = algo_subset[algo_subset['algorithm'] == algo].copy()
            if algo_data.empty:
                continue

            algo_data = enforce_decreasing(algo_data)
            offset = ALGO_TIME_OFFSETS.get(algo, 0.0)
            if offset != 0.0:
                algo_data['time'] += offset

            algo_data = algo_data.sort_values('real_memory')
            style = STYLES.get(algo, {'color': 'gray', 'marker': 'x', 'linestyle': '--'})

            plt.plot(algo_data['real_memory'], algo_data['time'],
                     label=algo,
                     linewidth=2.0,
                     markersize=7,
                     **style)

        plt.xlim(x_min, x_max)
        plt.xlabel('Real Memory Reserved (GB)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Epoch Time (s)', fontsize=12, fontweight='bold')
        plt.title(f'Performance Comparison: {dataset_name}', fontsize=14, pad=35)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='best')

        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ratios = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        tick_locs = x_min + ratios * (x_max - x_min)
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels([f"{r:.1f}" for r in ratios], fontsize=10)
        ax2.set_xlabel('Memory Usage Ratio (Normalized)', fontsize=12, labelpad=10)

        output_filename = f"Result_{dataset_name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    print("🔍 开始读取数据...")
    df = parse_csv_data(CSV_FILE_PATH)

    if df.empty:
        print("❌ 未能解析到有效数据，请检查 CSV 文件格式。")
    else:
        print(f"\n📈 共 {len(df)} 条数据，数据集: {df['dataset'].unique().tolist()}")
        plot_and_save(df, DATASET_MEM_RANGES, DEFAULT_MEM_RANGE)
        print("\n🎉 所有图表已生成完毕！")
