import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


# ==============================
# 🔧 配置区 —— 已按你要求修改
# ==============================
FOLDERS_TO_SCAN: List[str] = [
    "cricket_logs_checkmate",
    "cricket_logs_mimose",
    "cricket_logs_monet",
    "cricket_logs_oursILP",
]
# ==============================


def extract_scale_from_filename(filename: str) -> Optional[float]:
    """
    从文件名中智能提取 scale 值，支持多种命名风格：
      - *_1.2scale.log
      - *_scale1.5.log
      - *_s2.0.log
      - *_scale=0.8.log
      - *_0.5x.log
      - *_x2.log

    范围校验：仅接受 0.01 ~ 20.0（防误匹配编号）
    """
    stem = Path(filename).stem.lower()  # 去扩展名，转小写

    patterns = [
        r'(?:^|_)(\d+\.?\d*)\s*_*scale',          # _1.2scale
        r'(?:^|_)scale\s*_*?(\d+\.?\d*)',         # _scale1.2
        r'(?:^|_)s(\d+\.?\d*)',                   # _s1.5
        r'(?:^|_)scale\s*=\s*(\d+\.?\d*)',        # _scale=1.2
        r'(?:^|_)(\d+\.?\d*)x(?![a-z])',          # _0.5x, _2x （负向 lookahead 避免 xxx）
        r'(?:^|_)x(\d+\.?\d*)',                   # _x2, _x1.5
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            try:
                val = float(match.group(1))
                if 0.01 <= val <= 20.0:  # 合理 scale 范围
                    return val
            except (ValueError, TypeError, IndexError):
                continue
    return None


def _get_natural_sort_key(text: str) -> tuple:
    """
    将字符串转为可比较的自然序元组，例如：
        "run_1.2scale" → ('run_', 1.2, 'scale')
        "run_10scale"  → ('run_', 10.0, 'scale')
    """
    text = str(text).lower()
    parts = re.split(r'(\d+\.?\d*)', text)
    result = []
    for part in parts:
        if part.replace('.', '', 1).isdigit():
            try:
                result.append(float(part))
            except ValueError:
                result.append(part)
        else:
            result.append(part)
    return tuple(result)


def extract_info_from_log(file_path: Path) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """从日志提取：scale（从文件名）、max_reserved、allocated、avg_time"""
    filename = file_path.name
    scale = extract_scale_from_filename(filename)

    max_reserved_gb = None
    max_allocate_gb = None
    avg_time = None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}: {e}")
        return (filename, scale, max_reserved_gb, max_allocate_gb, avg_time)

    # 从后往前找最后一个 max_memory_reserved
    for line in reversed(lines):
        if "torch.cuda.max_memory_reserved()" in line:
            m = re.search(r'torch\.cuda\.max_memory_reserved\(\)\s*:\s*([\d.]+)', line)
            if m:
                try:
                    max_reserved_gb = float(m.group(1))
                except (ValueError, TypeError):
                    pass
                break

    # 从后往前找最后一个 max_memory_allocated
    for line in reversed(lines):
        if "torch.cuda.max_memory_allocated()" in line:
            m = re.search(r'torch\.cuda\.max_memory_allocated\(\)\s*:\s*([\d.]+)', line)
            if m:
                try:
                    max_allocate_gb = float(m.group(1))
                except (ValueError, TypeError):
                    pass
                break

    # 提取平均时间
    for line in lines:
        if "!!!采用策略后的几个epoch 平均时间:" in line:
            m = re.search(r'!!!采用策略后的几个epoch 平均时间:\s*([\d.]+)', line)
            if m:
                try:
                    avg_time = float(m.group(1))
                except (ValueError, TypeError):
                    pass
                break

    return (filename, scale, max_reserved_gb, max_allocate_gb, avg_time)


def scan_folder(folder: str) -> List[Tuple]:
    """扫描单个文件夹，返回 (filename, scale, reserved, allocated, time, folder_name)"""
    folder_path = Path(folder)
    results = []

    if not folder_path.exists():
        print(f"🔶 跳过不存在的路径: {folder_path}")
        return results

    if not folder_path.is_dir():
        print(f"🔶 跳过非目录路径: {folder_path}")
        return results

    log_files = list(folder_path.glob("*.log"))
    if not log_files:
        print(f"🔶 文件夹无 .log 文件: {folder_path}")
        return results

    print(f"📂 扫描 {folder_path.name} → {len(log_files)} 个日志")
    for log_file in log_files:
        try:
            info = extract_info_from_log(log_file)
            # 补全 folder 字段（确保是 str）
            folder_name = folder_path.name
            results.append((*info, folder_name))
            scale_display = f"{info[1]:.2f}" if info[1] is not None else "❓"
            print(f"   ✅ {log_file.name} → scale={scale_display}")
        except Exception as e:
            print(f"   ❌ 解析失败 {log_file.name}: {e}")

    return results


def main():
    print("🚀 开始聚合四个实验组的日志...\n")

    all_results = []
    for folder in FOLDERS_TO_SCAN:
        try:
            res = scan_folder(folder)
            all_results.extend(res)
        except Exception as e:
            print(f"💥 扫描 {folder} 时崩溃: {e}")

    if not all_results:
        print("❌ 未收集到任何日志数据。")
        return

    # ✅ 数据清洗：确保所有字段为基本类型（防 list / Path 混入）
    cleaned = []
    for row in all_results:
        if len(row) != 6:
            print(f"⚠️ 跳过异常行（长度≠6）: {row}")
            continue
        filename, scale, reserved, allocated, time_sec, folder = row
        # 转为纯净字符串（核心！防 unhashable list）
        filename = str(filename).strip() if filename is not None else ""
        folder = str(folder).strip() if folder is not None else "unknown"
        cleaned.append((filename, scale, reserved, allocated, time_sec, folder))

    # 构建 DataFrame
    df = pd.DataFrame(
        cleaned,
        columns=[
            "文件名",
            "scale",
            "max_memory_reserved (GB)",
            "max_memory_allocated (GB)",
            "策略后平均 epoch 时间 (s)",
            "来源文件夹"
        ]
    )

    # ✅ 安全自然排序（兼容所有 pandas 版本）
    df["_sort_folder"] = df["来源文件夹"].map(_get_natural_sort_key)
    df["_sort_filename"] = df["文件名"].map(_get_natural_sort_key)

    df = df.sort_values(
        by=["_sort_folder", "_sort_filename"],
        ascending=[True, True]
    ).reset_index(drop=True)

    df = df.drop(columns=["_sort_folder", "_sort_filename"])

    # 输出
    output_path = Path(__file__).parent.resolve() / "aggregated_log_summary.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"🎉 成功聚合 {len(df)} 条日志记录")
    print(f"📁 结果已保存至：\n{output_path.absolute()}")
    print("=" * 60)

    # 分组统计
    print("\n📊 实验组统计概览：")
    for folder, group in df.groupby("来源文件夹", sort=False):
        n = len(group)
        scale_vals = group["scale"].dropna()
        time_vals = group["策略后平均 epoch 时间 (s)"].dropna()
        mem_vals = group["max_memory_reserved (GB)"].dropna()

        print(f"\n📁 {folder}：{n} 个实验")
        if not scale_vals.empty:
            print(f"   • scale: {scale_vals.min():.2f} ~ {scale_vals.max():.2f}")
        if not time_vals.empty:
            print(f"   • 平均训练时间: {time_vals.mean():.1f} ± {time_vals.std():.1f} s")
        if not mem_vals.empty:
            print(f"   • 显存峰值 (reserved): {mem_vals.mean():.1f} ± {mem_vals.std():.1f} GB")


if __name__ == "__main__":
    main()