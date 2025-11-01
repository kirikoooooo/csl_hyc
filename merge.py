import os
import subprocess
import pandas as pd
from pathlib import Path

def merge_experiment_results(folder_list, csv_filename="log_summery.csv", output_file="merged_results.csv"):
    """
    对每个文件夹：
      1. 执行 log_summery.py
      2. 读取生成的 log_summery.csv
      3. 在数据前插入一行标注算法（文件夹名）
    最后合并所有结果。

    参数:
        folder_list (list of str): 要处理的文件夹路径列表。
        csv_filename (str): 每个文件夹中期望生成的 CSV 文件名。
        output_file (str): 最终合并输出的文件名。
    """
    all_data = []

    for folder in folder_list:
        folder_path = Path(folder).resolve()

        if not folder_path.is_dir():
            print(f"⚠️ 警告: 文件夹 {folder_path} 不存在，跳过。")
            continue

        script_path = folder_path / "log_extract.py"
        csv_path = folder_path / csv_filename

        # Step 1: 执行 log_summery.py
        if not script_path.exists():
            print(f"⚠️ 警告: {script_path} 不存在，跳过该文件夹。")
            continue

        print(f"🚀 正在执行: {script_path}")
        try:
            # 在该文件夹目录下运行脚本，确保相对路径正确
            result = subprocess.run(
                ["python", "log_extract.py"],
                cwd=folder_path,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            if result.returncode != 0:
                print(f"❌ 执行失败 ({folder}): {result.stderr}")
                continue
        except subprocess.TimeoutExpired:
            print(f"⏰ 执行超时 ({folder})，跳过。")
            continue
        except Exception as e:
            print(f"💥 执行异常 ({folder}): {e}")
            continue

        # Step 2: 检查 CSV 是否生成
        if not csv_path.exists():
            print(f"⚠️ 警告: 执行后未找到 {csv_path}，跳过。")
            continue

        # Step 3: 读取 CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ 读取 CSV 失败 ({csv_path}): {e}")
            continue

        if df.empty:
            print(f"⚠️ 警告: {csv_path} 为空，跳过。")
            continue

        # Step 4: 添加算法标注行
        algo_row = pd.DataFrame([{df.columns[0]: f"Algorithm: {folder_path.name}"}])
        for col in df.columns[1:]:
            algo_row[col] = pd.NA

        combined = pd.concat([algo_row, df], ignore_index=True)
        all_data.append(combined)

        # 可选：添加空行分隔
        empty_row = pd.DataFrame([[pd.NA] * len(df.columns)], columns=df.columns)
        all_data.append(empty_row)

    if not all_data:
        print("❌ 没有成功处理任何实验文件夹。")
        return

    # 合并并保存
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ 全部完成！合并结果已保存至: {output_file}")

# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    experiment_folders = [
        "cricket_logs_diff",
        "cricket_logs_ga",
        "cricket_logs_checkmate",
        "cricket_logs_monet",
        "cricket_logs_mimose"
    ]

    merge_experiment_results(
        folder_list=experiment_folders,
        csv_filename="log_summary.csv",
        output_file="merged_results.csv"
    )