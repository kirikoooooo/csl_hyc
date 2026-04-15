"""
从 PyTorch Profiler 导出的 Chrome Trace JSON 中解析 CUDA 显存采样与 record_function 区间，
计算各区间内峰值并绘图。输出默认写入项目 trace/ 目录。

用法:
  python profiler_trace_plot.py /path/to/chrome_trace.json
  python profiler_trace_plot.py trace.json -o ../trace --prefix CL. train.

依赖: matplotlib, numpy
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 与本项目 train.py / blocks.py 中 record_function 命名一致的可选前缀
DEFAULT_SPAN_PREFIXES: Tuple[str, ...] = (
    "CL.",
    "train.",
    "model/",
    "mix/",
    "shapelets/",
)


@dataclass
class MemorySample:
    """单点显存采样（Chrome trace 时间戳为微秒）。"""
    ts_us: float
    allocated_bytes: float


@dataclass
class DurationSpan:
    """ph=='X' 的区间事件（record_function 等）。"""
    name: str
    ts_us: float
    dur_us: float

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


@dataclass
class SpanPeakResult:
    span_name: str
    start_us: float
    end_us: float
    peak_gb: float
    peak_t_us: float


def load_chrome_trace(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_trace_events(trace: dict) -> List[dict]:
    return list(trace.get("traceEvents") or [])


def extract_cuda_memory_samples(
    events: Iterable[dict],
    require_device_cuda: bool = True,
) -> List[MemorySample]:
    """
    解析参考格式: name == "[memory]", args 含 Total Allocated;
    Device Type == 1 表示 CUDA（部分版本可能无此字段，可关 require_device_cuda）。
    """
    out: List[MemorySample] = []
    for evt in events:
        if evt.get("name") != "[memory]":
            continue
        args = evt.get("args") or {}
        if require_device_cuda and "Device Type" in args and args.get("Device Type") != 1:
            continue
        ta = args.get("Total Allocated")
        if ta is None:
            continue
        ts = float(evt.get("ts", 0))
        out.append(MemorySample(ts_us=ts, allocated_bytes=float(ta)))
    out.sort(key=lambda s: s.ts_us)
    return out


def default_span_name_filter(name: str) -> bool:
    if not name or name == "[memory]":
        return False
    if name.startswith("ProfilerStep"):
        return False
    if name.startswith("aten::") or name.startswith("cuda"):
        return False
    return any(name.startswith(p) for p in DEFAULT_SPAN_PREFIXES)


def extract_duration_spans(
    events: Iterable[dict],
    name_filter: Optional[Callable[[str], bool]] = None,
) -> List[DurationSpan]:
    """收集 ph=='X' 且带 dur 的区间，用于对齐 record_function 标签。"""
    filt = name_filter or default_span_name_filter
    spans: List[DurationSpan] = []
    for evt in events:
        if evt.get("ph") != "X":
            continue
        dur = evt.get("dur")
        if dur is None:
            continue
        name = evt.get("name") or ""
        if not filt(name):
            continue
        ts = float(evt.get("ts", 0))
        spans.append(DurationSpan(name=name, ts_us=ts, dur_us=float(dur)))
    spans.sort(key=lambda s: (s.ts_us, s.name))
    return spans


def samples_to_arrays_ms_gb(samples: Sequence[MemorySample]) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.array([]), np.array([])
    t = np.array([s.ts_us for s in samples], dtype=np.float64)
    v = np.array([s.allocated_bytes / (1024 ** 3) for s in samples], dtype=np.float64)
    return t, v


def peak_in_interval(
    t_us: np.ndarray,
    v_gb: np.ndarray,
    start_us: float,
    end_us: float,
) -> Tuple[float, float]:
    """区间内最大显存及对应时间（微秒）；无采样则返回 (0.0, start_us)。"""
    if t_us.size == 0:
        return 0.0, start_us
    mask = (t_us >= start_us) & (t_us <= end_us)
    if not np.any(mask):
        return 0.0, start_us
    idx = int(np.argmax(v_gb[mask]))
    sub_t = t_us[mask]
    sub_v = v_gb[mask]
    return float(sub_v[idx]), float(sub_t[idx])


def compute_span_peaks(
    samples: Sequence[MemorySample],
    spans: Sequence[DurationSpan],
) -> List[SpanPeakResult]:
    t_us, v_gb = samples_to_arrays_ms_gb(samples)
    results: List[SpanPeakResult] = []
    for sp in spans:
        peak_gb, peak_t = peak_in_interval(t_us, v_gb, sp.ts_us, sp.end_us)
        results.append(
            SpanPeakResult(
                span_name=sp.name,
                start_us=sp.ts_us,
                end_us=sp.end_us,
                peak_gb=peak_gb,
                peak_t_us=peak_t,
            )
        )
    return results


def aggregate_peaks_by_name(span_peaks: Sequence[SpanPeakResult]) -> dict:
    """同名多段区间：保留全局最大峰值。"""
    best: dict = {}
    for r in span_peaks:
        prev = best.get(r.span_name)
        if prev is None or r.peak_gb > prev["peak_gb"]:
            best[r.span_name] = {
                "peak_gb": r.peak_gb,
                "peak_t_us": r.peak_t_us,
                "start_us": r.start_us,
                "end_us": r.end_us,
            }
    return best


def plot_memory_with_spans(
    samples: Sequence[MemorySample],
    spans: Sequence[DurationSpan],
    span_peaks: Sequence[SpanPeakResult],
    output_path: str,
    title: str = "CUDA memory vs record_function spans",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    t_us, v_gb = samples_to_arrays_ms_gb(samples)
    if t_us.size == 0:
        raise ValueError("无显存采样点：请确认 trace 由 profile_memory=True 导出且含 [memory] 事件")

    t0 = float(np.min(t_us))
    if spans:
        t0 = min(t0, min(s.ts_us for s in spans))
    rel_s = (t_us - t0) / 1e6
    rel_peak_t = [(r.peak_t_us - t0) / 1e6 for r in span_peaks]
    rel_peak_v = [r.peak_gb for r in span_peaks]

    plt.figure(figsize=(14, 6))
    plt.step(rel_s, v_gb, where="post", color="#2E86AB", lw=2, label="CUDA allocated (trace)")

    cmap = plt.cm.tab20(np.linspace(0, 1, max(20, len(spans) + 1)))
    for i, sp in enumerate(spans):
        s0 = (sp.ts_us - t0) / 1e6
        s1 = (sp.end_us - t0) / 1e6
        plt.axvspan(s0, s1, color=cmap[i % len(cmap)], alpha=0.12)

    # 标注各区间实例的峰值点（同名多次会出现多个点）
    if span_peaks:
        plt.scatter(rel_peak_t, rel_peak_v, c="#E94F37", s=36, zorder=5, label="Peak in span")

    g_idx = int(np.argmax(v_gb))
    g_t = rel_s[g_idx]
    g_v = float(v_gb[g_idx])
    plt.scatter([g_t], [g_v], c="gold", s=140, zorder=6, marker="*")
    plt.annotate(
        f"global {g_v:.3f} GB",
        xy=(g_t, g_v),
        xytext=(g_t + (rel_s[-1] - rel_s[0]) * 0.02, g_v + 0.02),
        arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.2),
        fontsize=10,
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Allocated (GB)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_prefix_filter(prefixes: Sequence[str]) -> Callable[[str], bool]:
    pfx = tuple(prefixes)

    def filt(name: str) -> bool:
        if not name or name == "[memory]":
            return False
        if name.startswith("ProfilerStep"):
            return False
        return any(name.startswith(p) for p in pfx)

    return filt


def process_trace_file(
    trace_json_path: str,
    output_dir: str,
    span_prefixes: Optional[Sequence[str]] = None,
    require_device_cuda: bool = True,
    basename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    解析 trace，写 PNG 与峰值摘要 TXT。返回 (png_path, summary_path)。
    """
    trace = load_chrome_trace(trace_json_path)
    events = iter_trace_events(trace)
    samples = extract_cuda_memory_samples(events, require_device_cuda=require_device_cuda)
    if not samples:
        raise ValueError(
            "未解析到任何 [memory] 显存采样。请使用 profile_memory=True 导出 trace；"
            "若仍为空可尝试 --no-device-filter。"
        )
    name_filter = (
        make_prefix_filter(span_prefixes)
        if span_prefixes is not None
        else default_span_name_filter
    )
    spans = extract_duration_spans(events, name_filter=name_filter)
    span_peaks = compute_span_peaks(samples, spans)
    by_name = aggregate_peaks_by_name(span_peaks)

    stem = basename or os.path.splitext(os.path.basename(trace_json_path))[0]
    safe = re.sub(r"[^\w\-.]+", "_", stem)[:120]
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{safe}_memory_spans.png")
    txt_path = os.path.join(output_dir, f"{safe}_span_peaks.txt")

    t0_mem = min(s.ts_us for s in samples) if samples else 0.0

    lines = [
        f"trace: {trace_json_path}",
        f"memory samples: {len(samples)}, spans: {len(spans)}",
        "",
        "--- per span instance (peak in [start,end]) ---",
    ]
    for r in span_peaks:
        lines.append(
            f"{r.span_name}\tpeak={r.peak_gb:.6f} GB\t@ {(r.peak_t_us - t0_mem) / 1e6:.6f}s "
            f"(span {(r.start_us - t0_mem) / 1e6:.6f}s .. {(r.end_us - t0_mem) / 1e6:.6f}s)"
        )
    lines.append("")
    lines.append("--- max peak by span name ---")
    for name in sorted(by_name.keys()):
        b = by_name[name]
        lines.append(f"{name}\t{b['peak_gb']:.6f} GB")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plot_memory_with_spans(
        samples,
        spans,
        span_peaks,
        png_path,
        title=f"Memory vs spans — {stem}",
    )

    return png_path, txt_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot CUDA memory from PyTorch Chrome trace + record_function spans")
    p.add_argument("trace_json", help="export_chrome_trace 生成的 .json 路径")
    p.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "trace"),
        help="输出目录（默认 csl_hyc/trace）",
    )
    p.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="只保留 name 以这些前缀开头的 X 区间；不传则用项目默认 CL. train. model/ mix/ shapelets/",
    )
    p.add_argument(
        "--no-device-filter",
        action="store_true",
        help="不强制 Device Type==1（部分 trace 格式不同）",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    prefixes = tuple(args.prefix) if args.prefix else None
    png, txt = process_trace_file(
        args.trace_json,
        args.output_dir,
        span_prefixes=prefixes,
        require_device_cuda=not args.no_device_filter,
    )
    print(f"Wrote {png}\n{txt}")


if __name__ == "__main__":
    main()
