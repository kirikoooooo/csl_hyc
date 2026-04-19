import torch
import time
import torch.nn as nn
import torch.nn.functional as F

# ====================== 环境与配置 ======================
device = torch.device("cuda")
print(f"✅ 使用设备: {device} | GPU: {torch.cuda.get_device_name(device)}")
torch.cuda.empty_cache()

# 严谨性：禁用 TF32，保证计算路径一致
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ====================== 计时器（融合优点）======================

def benchmark_event(func, warmup=5, repeat=10):
    """方案1：torch.cuda.Event + 同步（官方推荐）"""
    # 充分预热
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(repeat):
        torch.cuda.synchronize()
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)

def benchmark_time(func, warmup=5, repeat=10):
    """方案2：time.perf_counter() + 同步（专家级精度）"""
    # 充分预热
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()  # ✅ 用 perf_counter，专家级选择
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)

# ====================== 严谨的 Workload（类封装 + 全局变量结合）======================

# 1. 矩阵乘法（单次计算，靠 repeat 平均）
class MatmulBench:
    def __init__(self):
        self.a = torch.randn(4096, 4096, device=device)
        self.b = torch.randn(4096, 4096, device=device)
    def __call__(self):
        torch.matmul(self.a, self.b)

# 2. CNN（类封装）
class CNNBench:
    def __init__(self):
        self.x = torch.randn(32, 3, 256, 256, device=device)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1).to(device)
    def __call__(self):
        x = self.conv1(self.x)
        x = F.relu(x)
        x = self.conv2(x)

# 3. Transformer
class TransformerBench:
    def __init__(self):
        self.x = torch.randn(64, 256, 256, device=device)
        self.layer = nn.TransformerEncoderLayer(256, 8, batch_first=True).to(device)
    def __call__(self):
        self.layer(self.x)

# 4. FP16 矩阵乘法
class FP16MatmulBench:
    def __init__(self):
        self.a = torch.randn(6144, 6144, device=device, dtype=torch.float16)
        self.b = torch.randn(6144, 6144, device=device, dtype=torch.float16)
    def __call__(self):
        torch.matmul(self.a, self.b)

# 5. cdist（重点）
class CdistBench:
    def __init__(self):
        self.x = torch.randn(8192, 128, device=device)
        self.y = torch.randn(8192, 128, device=device)
    def __call__(self):
        torch.cdist(self.x, self.y)

# 6. 前向+反向传播（专家级数值处理）
class ForwardBackwardBench:
    def __init__(self):
        self.w = torch.randn(2048, 2048, device=device)
    def __call__(self):
        # 每次创建新的 x，避免梯度累积；数值缩放避免爆炸
        x = torch.randn(2048, 2048, device=device, requires_grad=True) / 100.0
        y = torch.matmul(x, self.w)
        loss = y.sum()
        loss.backward()

# ====================== 运行对比 ======================
print("\n" + "="*80)
print("📊 终极完美版：Event+同步  vs  Time+同步")
print("="*80)

benchmarks = [
    ("矩阵乘法", MatmulBench()),
    ("CNN卷积", CNNBench()),
    ("Transformer", TransformerBench()),
    ("FP16矩阵", FP16MatmulBench()),
    ("cdist距离", CdistBench()),
    ("前向+反向传播", ForwardBackwardBench()),
]

for name, bench in benchmarks:
    torch.cuda.empty_cache()
    t_evt = benchmark_event(bench)
    t_tim = benchmark_time(bench)
    diff = t_tim - t_evt

    print(f"\n🔹 {name}")
    print(f"  Event+同步：{t_evt:>8.2f} ms  (精准 GPU 时间)")
    print(f"  Time+同步：{t_tim:>8.2f} ms  (含 CPU 调度开销)")
    print(f"  差异(Time-Event)：{diff:>6.2f} ms")

print("\n" + "="*80)
print("📌 总结：")
print("  1. 使用了 time.perf_counter() 保证纳秒级精度")
print("  2. 充分预热 5 次，保证 GPU 状态稳定")
print("  3. 禁用 TF32，保证计算路径一致")
print("  4. 类封装 + 全局变量结合，代码整洁且严谨")
print("  5. 反向传播做了数值缩放，避免梯度爆炸")
print("="*80)