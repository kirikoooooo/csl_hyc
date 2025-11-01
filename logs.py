block_forward_stats_by_type = {
    "euclidean": {},  # shapelet_length: {"forward_total_time": float, "forward_calls": int, ...}
    "cosine": {},
    "cross": {}
}

# 记录全局反向传播峰值显存
global_backward_peak_mem = 0

global_backward_total_time = 0.0

global_backward_b =0.0

# 记录全局每个模块 forward 前已使用的显存（仅一次）
global_pre_forward_mem = None

# 第一个 epoch 的第一次迭代是否跳过记录（用于排除加载数据时间）
skip_first_forward = True

peak_memory_log = []

global_iter_count = 0

cdist_euclidean_mem = None  # 仅记录一次 torch.cdist 的显存消耗（单位：Byte）

x = [1] * 33

# 初始化为空，在训练前设置为全 shapelet 长度
euclidean_checkpoint_shapelet_lengths = []
cosine_checkpoint_shapelet_lengths = []