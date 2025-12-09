import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# ==========================================
# 🔧 配置
# ==========================================

STYLES = {
    'Checkmate':  {'color': '#e74c3c', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 5},
    'Monet':      {'color': '#f1c40f', 'marker': 's', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 5},
    'Mimose':     {'color': '#3498db', 'marker': '^', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 5},
    'Ours (ILP)': {'color': '#2ecc71', 'marker': '*', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 8},
}

# 增加 X 轴范围配置
DEFAULT_X_LIM = (0.7, 1.5)

# ==========================================
# 🔍 解析 dataset & algorithm
# ==========================================
def parse_dataset_and_algo(filename: str):
    fname = filename.strip()
    if fname.startswith("Cricketcheckmate"):
        return "Cricket", "Checkmate"
    elif fname.startswith("Cricketmimose"):
        return "Cricket", "Mimose"
    elif fname.startswith("Cricketmonet"):
        return "Cricket", "Monet"
    elif fname.startswith("CricketoursILP"):
        return "Cricket", "Ours (ILP)"
    else:
        for prefix in ["Cricket", "DuckDuckGeese", "MotorImagery", "PEMS"]:
            if fname.startswith(prefix):
                algo_part = fname[len(prefix):].lower()
                if "checkmate" in algo_part:
                    return prefix, "Checkmate"
                elif "mimose" in algo_part:
                    return prefix, "Mimose"
                elif "monet" in algo_part:
                    return prefix, "Monet"
                elif "ours" in algo_part or "ilp" in algo_part:
                    return prefix, "Ours (ILP)"
        return "Unknown", "Unknown"

# ==========================================
# 📊 绘图 (线性 Y 轴)
# ==========================================
def plot_scale_vs_time(df, x_lim_range=DEFAULT_X_LIM):
    """
    绘制 Scale Factor vs Average Epoch Time (线性 Y 轴)。
    此函数默认使用 Matplotlib 的自动 Y 轴范围。
    """
    df = df.dropna(subset=['scale', '策略后平均 epoch 时间 (s)']).copy()
    parsed = df['文件名'].apply(parse_dataset_and_algo)
    df['dataset'] = [x[0] for x in parsed]
    df['algorithm'] = [x[1] for x in parsed]
    valid_algos = ['Checkmate', 'Monet', 'Mimose', 'Ours (ILP)']
    df = df[df['algorithm'].isin(valid_algos)]
    df = df[df['dataset'] != "Unknown"]

    print(f"📊 有效数据: {len(df)} 条 | Dataset: {df['dataset'].unique()}")

    for dataset in sorted(df['dataset'].unique()):
        ds_df = df[df['dataset'] == dataset]
        plt.figure(figsize=(8, 5))
        plt.style.use('seaborn-v0_8-whitegrid')

        for algo in valid_algos:
            algo_df = ds_df[ds_df['algorithm'] == algo].copy()
            if algo_df.empty:
                continue
            algo_df = algo_df.sort_values('scale')
            plt.plot(
                algo_df['scale'],
                algo_df['策略后平均 epoch 时间 (s)'],
                label=algo,
                **STYLES[algo]
            )

        plt.xlabel('Scale Factor', fontsize=12, fontweight='bold')
        plt.ylabel('Average Epoch Time (s)', fontsize=12, fontweight='bold')
        plt.title(f'{dataset}: Time vs Scale', fontsize=13, pad=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='best')

        # 应用配置的 X 轴范围
        plt.xlim(x_lim_range)

        out_file = f"ScaleTime_{dataset}_Full.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {out_file}")

# ==========================================
# 📊 绘图 (Log Y 轴 - 自适应)
# ==========================================
def plot_scale_vs_time_v2(df, x_lim_range=DEFAULT_X_LIM):
    """
    绘制 Scale Factor vs Average Epoch Time (Log Y 轴)。
    Y 轴范围和刻度现已根据当前数据集自适应调整。
    """
    df = df.dropna(subset=['scale', '策略后平均 epoch 时间 (s)']).copy()
    parsed = df['文件名'].apply(parse_dataset_and_algo)
    df['dataset'] = [x[0] for x in parsed]
    df['algorithm'] = [x[1] for x in parsed]
    valid_algos = ['Checkmate', 'Monet', 'Mimose', 'Ours (ILP)']
    df = df[df['algorithm'].isin(valid_algos)]
    df = df[df['dataset'] != "Unknown"]

    print(f"📊 有效数据: {len(df)} 条 | Dataset: {df['dataset'].unique()}")

    for dataset in sorted(df['dataset'].unique()):
        ds_df = df[df['dataset'] == dataset].copy()
        if ds_df.empty: continue

        plt.figure(figsize=(8.5, 5.5))
        plt.style.use('seaborn-v0_8-whitegrid')

        # === 🎯 针对当前数据集计算 Y 轴范围 ===
        y_vals = ds_df['策略后平均 epoch 时间 (s)']
        y_min = y_vals.min() * 0.95
        y_max = y_vals.max() * 1.05

        # 动态计算刻度，保留 2 位小数
        num_ticks = 6
        y_ticks = np.linspace(y_min, y_max, num_ticks)
        y_tick_labels = [f"{t:.2f}" for t in y_ticks]

        for algo in valid_algos:
            algo_df = ds_df[ds_df['algorithm'] == algo].copy()
            if algo_df.empty:
                continue
            algo_df = algo_df.sort_values('scale')
            plt.plot(
                algo_df['scale'],
                algo_df['策略后平均 epoch 时间 (s)'],
                label=algo,
                **STYLES[algo]
            )

        plt.xlabel('Scale Factor', fontsize=13, fontweight='bold')
        plt.ylabel('Average Epoch Time (s)', fontsize=13, fontweight='bold')
        plt.title(f'{dataset}: Time vs Scale (Log Scale - Adaptive)',
                  fontsize=14, pad=15, fontweight='bold')

        plt.yscale('log')
        plt.ylim(y_min, y_max)
        plt.yticks(y_ticks, y_tick_labels) # 应用动态刻度

        plt.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.7)
        plt.grid(True, which="minor", linestyle='--', linewidth=0.5, alpha=0.4)

        plt.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
        plt.xlim(x_lim_range)

        out_file = f"ScaleTime_{dataset}_LogScale_Adaptive.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存（对数坐标，Y轴自适应）: {out_file}")

# ==========================================
# 📊 绘图 (Log-Speed Y 轴, Y 轴反转 - 自适应)
# ==========================================
def plot_scale_vs_time_v3(df, x_lim_range=DEFAULT_X_LIM):
    """
    绘制 Scale Factor vs Average Epoch Time (Log-Speed Y 轴, Y 轴反转)。
    Y 轴范围和刻度现已根据当前数据集自适应调整。
    """
    df = df.dropna(subset=['scale', '策略后平均 epoch 时间 (s)']).copy()
    parsed = df['文件名'].apply(parse_dataset_and_algo)
    df['dataset'] = [x[0] for x in parsed]
    df['algorithm'] = [x[1] for x in parsed]
    valid_algos = ['Checkmate', 'Monet', 'Mimose', 'Ours (ILP)']
    df = df[df['algorithm'].isin(valid_algos)]
    df = df[df['dataset'] != "Unknown"]

    print(f"📊 有效数据: {len(df)} 条 | Dataset: {df['dataset'].unique()}")

    def forward(y):
        return np.log(1.0 / y)  # log-speed scale

    def inverse(y):
        return 1.0 / np.exp(y)

    for dataset in sorted(df['dataset'].unique()):
        ds_df = df[df['dataset'] == dataset].copy()
        if ds_df.empty: continue

        plt.figure(figsize=(9, 5.8))
        plt.style.use('seaborn-v0_8-whitegrid')
        ax = plt.gca()

        ax.set_yscale('function', functions=(forward, inverse))

        for algo in valid_algos:
            algo_df = ds_df[ds_df['algorithm'] == algo].copy()
            if algo_df.empty:
                continue
            algo_df = algo_df.sort_values('scale')
            plt.plot(
                algo_df['scale'],
                algo_df['策略后平均 epoch 时间 (s)'],
                label=algo,
                **STYLES[algo]
            )

        # === 🎯 针对当前数据集计算 Y 轴范围和刻度 ===
        y_vals = ds_df['策略后平均 epoch 时间 (s)']
        y_min_data, y_max_data = y_vals.min(), y_vals.max()

        # 增加 2% 边距
        y_min_plot = y_min_data * 0.98
        y_max_plot = y_max_data * 1.02

        # 动态生成刻度：使用 log-space 确保在反转轴上分布均匀
        num_ticks = 10 # 更多的刻度来保证精度
        # 在 log(1/y) 空间上线性分布
        log_y_ticks_trans = np.linspace(forward(y_max_plot), forward(y_min_plot), num_ticks)
        # 转换回原始 y 值
        y_ticks = inverse(log_y_ticks_trans)

        # 格式化刻度标签，保留 2 位小数
        y_tick_labels = [f"{y:.2f}" for y in y_ticks]

        # ——————— 坐标轴美化 ———————
        plt.xlabel('Scale Factor', fontsize=13, fontweight='bold')
        plt.ylabel('Average Epoch Time (s)', fontsize=13, fontweight='bold')
        plt.title(f'{dataset}: Time vs Scale (log-Speed Scale - Adaptive)',
                  fontsize=14, pad=15, fontweight='bold')

        plt.yticks(y_ticks, y_tick_labels)
        plt.minorticks_off()

        plt.grid(True, which="major", linestyle='-', linewidth=0.9, alpha=0.75)
        plt.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
        plt.xlim(x_lim_range)
        plt.ylim(y_min_plot, y_max_plot) # 必须在反转前设置范围

        # 翻转 Y 轴！让小 time 在上方（符合“越快越高”的直觉）
        ax.invert_yaxis()

        out_file = f"ScaleTime_{dataset}_LogSpeed_Adaptive.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=350, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存（log-speed scale + Y轴反转，Y轴自适应）: {out_file}")

# ==========================================
# 🚀 主程序（从外部 CSV 文件读取）
# ==========================================
def main():
    # 🔴 请将 'data.csv' 替换为您实际的 CSV 文件路径
    csv_file = "aggregated_log_summary.csv"  # ←←← 修改这里！

    # 🔧 X 轴范围配置
    # (Scale Factor 的最小值, Scale Factor 的最大值)
    x_axis_range = DEFAULT_X_LIM

    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: '{csv_file}'")
        print("💡 请确保 CSV 文件与脚本在同一目录，或使用绝对路径。")
        return

    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功加载 {len(df)} 行数据（来自文件: {csv_file}）")
    except Exception as e:
        print(f"❌ 读取 CSV 文件失败: {e}")
        return

    # 检查必要列
    required_cols = ["文件名", "scale", "策略后平均 epoch 时间 (s)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ 缺少列: {missing}")
        return

    # 传递 X 轴范围配置，Y 轴范围在绘图函数内部自适应
    plot_scale_vs_time(df, x_lim_range=x_axis_range)
    plot_scale_vs_time_v2(df, x_lim_range=x_axis_range)
    plot_scale_vs_time_v3(df, x_lim_range=x_axis_range)

    print("\n🎉 绘图完成！新的自适应 Y 轴曲线图已保存。")


if __name__ == "__main__":
    main()