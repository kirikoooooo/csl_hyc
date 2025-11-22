import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


# ============================
# 提取数据
# 格式：{alg_name: (mem_limits, times)}
data_duck = {
    "checkmate": ([10,12,14,16,18,20,22], [6.0214,5.8462,5.9081,5.5814,5.5314,5.3853,5.2724]),
    "monet":     ([12,14,16,18,20,22],     [5.8208,5.6774,5.5842,5.5408,5.4820,5.2764]), # 10GB缺失
    "mimose":    ([10,12,14,16,18,20,22], [6.1300,6.0025,6.0009,5.6742,5.5517,5.4807,5.3658]),
    "oursILP":   ([10,12,14,16,18,20,22], [6.1458,6.1566,5.8535,5.6378,5.5267,5.2325,5.0880])
}
data_cricket = {
    "checkmate": ([8,10,12,14,16,18], [0.5354,0.4975,0.4777,0.4538,0.4550,0.4109]),
    "monet":     ([8,10,12,14,16,18], [0.5152,0.5129,0.4858,0.4462,0.4447,0.4094]),
    "mimose":    ([8,10,12,14,16,18], [0.5222,0.4903,0.4771,0.4614,0.4617,0.4189]),
    "oursILP":   ([8,10,12,14,16,18], [0.5209,0.4754,0.4511,0.4579,0.4163,0.4144])
}


data = data_duck

plt.figure(figsize=(10,6))
for alg, (xs, ys) in data.items():
    plt.plot(xs, ys, marker='o', label=alg)

plt.xlabel('memory budget (GB)', fontsize=12)
plt.ylabel('epoch time cost (s)', fontsize=12)
plt.title('compare 5 algorithm in duck 8bs', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xticks([10,12,14,16,18,20,22])
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图像已保存为 'algorithm_comparison.png'")