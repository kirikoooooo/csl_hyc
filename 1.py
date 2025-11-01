
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
# # 前向传播时间（x）和反向传播时间（y）
# x = np.array([0.0595, 0.0613, 0.0627, 0.0624, 0.0606, 0.0646, 0.0611, 0.0633, 0.0630, 0.0602, 0.0608, 0.0622, 0.0611]).reshape(-1, 1)
# y = np.array([0.1706, 0.1787, 0.1729, 0.1788, 0.1715, 0.1682, 0.1711, 0.1707, 0.1675, 0.1772, 0.1780, 0.1659, 0.1712])
#
# # 线性回归拟合
# model = LinearRegression()
# model.fit(x, y)
# y_pred_linear = model.predict(x)
#
# # 多项式回归拟合（二次）
# poly = PolynomialFeatures(degree=2)
# x_poly = poly.fit_transform(x)
# model_poly = LinearRegression(positive=True)
# model_poly.fit(x_poly, y)
# x_fit = np.linspace(min(x)[0], max(x)[0], 100).reshape(-1, 1)
# y_pred_poly = model_poly.predict(poly.transform(x_fit))
#
# # 画图
# plt.figure(figsize=(8, 5))
# plt.scatter(x, y, label="实际数据", color="blue")
# plt.plot(x, y_pred_linear, label="线性拟合", color="green", linestyle="--")
# plt.plot(x_fit, y_pred_poly, label="二次多项式拟合", color="red")
# plt.xlabel("前向传播时间")
# plt.ylabel("反向传播时间")
# plt.title("前向传播时间 vs 反向传播时间 拟合曲线")
# plt.legend()
# plt.grid(True)
# plt.show()
# # 打印线性拟合函数：y = a * x + b
# a = model.coef_[0]
# b = model.intercept_
# print(f"线性拟合函数：y = {a:.6f} * x + {b:.6f}")
#
# # 打印二次多项式拟合函数：y = a * x^2 + b * x + c
# a2 = model_poly.coef_[2]
# b2 = model_poly.coef_[1]
# c2 = model_poly.intercept_
# print(f"二次多项式拟合函数：y = {a2:.6f} * x² + {b2:.6f} * x + {c2:.6f}")
#
# # 拟合函数的系数
# (model.coef_[0], model.intercept_), model_poly.coef_, model_poly.intercept_
# import numpy as np
#
# # 前向传播时间（x）和反向传播时间（y）
# x = np.array([
#     0.0429, 0.0423, 0.0426, 0.0433, 0.0425, 0.0431, 0.0422,
#     0.0435, 0.0424, 0.0436, 0.0424, 0.0421, 0.0447
# ])
# y = np.array([
#     0.1202, 0.1251, 0.1252, 0.1232, 0.1248, 0.1228, 0.1248,
#     0.1199, 0.1247, 0.1229, 0.1245, 0.1331, 0.1221
# ])
#
# # 拟合 y ≈ k * x + c，其中 k 表示“几倍”，c 是常数项
# k = np.mean(y / x)
# c = np.mean(y - k * x)
# print(f"线性拟合函数：y = {k:.6f} * x + {c:.6f}")
# 模型函数
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
#
# # 数据（前向传播时间 x 和反向传播时间 y）
# x = np.array([0.4297, 0.4734, 0.4730, 0.4739, 0.4750, 0.4316,
#     0.4742, 0.4322, 0.4320, 0.4320, 0.4333, 0.4752])
# y = np.array([0.8645, 0.9321, 0.9515, 0.9325, 0.9299, 0.8696,
#     0.9349, 0.8693, 0.8494, 0.8539, 0.8625, 0.9370])
# def linear_model(x, a, b):
#     return a * x + b
#
# # 设置参数边界：a ∈ [1, 10]，b ∈ [-1, 1]
# popt, _ = curve_fit(linear_model, x, y, bounds=([2, -1], [10, 1]))
# a, b = popt
#
# # 打印拟合公式
# print(f"拟合公式：y = {a:.4f} * x + {b:.4f}")
#
# # 拟合曲线
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = linear_model(x_fit, a, b)
#
# # 可视化
# plt.figure(figsize=(8, 5))
# plt.scatter(x, y, color='blue', label='数据点')
# plt.plot(x_fit, y_fit, color='red', label=f'拟合：y = {a:.4f}x + {b:.4f}')
# plt.xlabel("前向传播时间")
# plt.ylabel("反向传播时间")
# plt.title("反向传播时间 vs 前向传播时间")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# import torch
# import torch.nn
# import torch.optim
# import torch.profiler
# import torch.utils.data
# import torchvision.datasets
# import torchvision.models
# import torchvision.transforms as T
# transform = T.Compose(
#     [T.Resize(224),
#      T.ToTensor(),
#      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
# device = torch.device("cuda:0")
# model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
# criterion = torch.nn.CrossEntropyLoss().cuda(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model.train()
# def train(data):
#     inputs, labels = data[0].to(device=device), data[1].to(device=device)
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# def train(data):
#     inputs, labels = data[0].to(device=device), data[1].to(device=device)
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
# ) as prof:
#     for step, batch_data in enumerate(train_loader):
#         prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
#         if step >= 1 + 1 + 3:
#             break
#         train(batch_data)



# x[i] = 0 or 1，表示模块 i 是否使用 checkpoint，i 从 1 到 32
x = [0] * 33  # 下标从1开始，x[0]不用
# 示例：x[1] = 1 表示第1个模块不使用 checkpoint，会保留最终显存

# 模块类型判断
def is_E_module(i):
    return 1 <= i <= 8 or 17 <= i <= 24

def is_C_module(i):
    return 9 <= i <= 16 or 25 <= i <= 32


# 显存峰值函数 P(i)
def get_peak_memory(i):
    if is_E_module(i):  # E模块
        return get_S_e((i - 1) % 8 + 1)
    elif is_C_module(i): # C模块
        return get_S_c((i - 9) % 8 + 1)
    else:
        raise ValueError(f"Invalid module index {i}")

# 前向传播最终保留显存 F(i)
def get_final_memory(i):
    if 1 <= i <= 8 or 17 <= i <= 24:  # E模块
        return x[i] * get_M_e((i - 1) % 8 + 1)
    elif 9 <= i <= 16 or 25 <= i <= 32:  # C模块
        return x[i] * get_M_c((i - 9) % 8 + 1)
    else:
        raise ValueError(f"Invalid module index {i}")

# 反向传播中释放的显存
def get_released_memory(i):
    if x[i] == 0:
        return 0
    if is_E_module(i):
        return get_M_e((i - 1) % 8 + 1)
    elif is_C_module(i):
        return get_M_c((i - 9) % 8 + 1)

# 反向传播的峰值显存
def get_backward_peak(i):
    if is_E_module(i):
        return back_peek/forward_peek * get_S_e((i - 1) % 8 + 1)
    elif is_C_module(i):
        return get_S_c((i - 9) % 8 + 1)

# 计算每个时间段的显存峰值 K_t
def compute_total_memory_peaks():
    K = [0] * 65  # K[0]不用
    cumulative_final = 0
    forward_K = []

    # 1. 前向传播 t=1~32
    for t in range(1, 33):
        peak = get_peak_memory(t)
        K[t] = p + peak + cumulative_final
        cumulative_final += get_final_memory(t)
        forward_K.append(K[t])

    # 2. 反向传播 t=33~64，对应模块 i = 65 - t
    total_final = sum(get_final_memory(j) for j in range(1, 33))
    cumulative_release = 0
    for t in range(33, 65):
        i = 65 - t  # 正在反向传播的模块
        backward_peak = get_backward_peak(i)
        released = get_released_memory(i)
        K[t] = backward_peak + total_final - cumulative_release
        cumulative_release += released

    return K

def compute_overall_peak():
    K = compute_total_memory_peaks()
    return max(K[1:])  # 排除 K[0]


# 你可以根据实际情况填入数值
def get_S_e(i):  # E模块的前向峰值显存
    return [0, 76116480, 133285376, 177382400, 202801664, 209345024, 204210688, 178998784, 135731712][i]  # i ∈ [1,8]
    # DD
    # return [0, 456920576,791817216,1034990592,1183666176,1235357696,1195588096,1062126592,833251840][i]

#     return [0, 192636416,
# 330189312,
# 432114176,
# 500934656,
# 525537792,
# 513451008,
# 462955520,
# 369516544][i]

def get_M_e(i):  # E模块的最终显存
    return [0, 28307016, 47227064, 60799544, 68842424, 71357608, 68387368, 59887528, 45858088][i]
    # DD
    # return [0,217589184,327986784,407008224,454653504,470922624,455815584,409332384,331473024][i]
#     return [0, 114713600,
# 159044096,
# 192308224,
# 213279232,
# 219998208,
# 215776768,
# 196762112,
# 165045248][i]

def get_S_c(i):  # C模块的前向峰值显存
    return [0, 49513984, 88092672, 117601792, 134355968, 138527744, 134310400, 117510144, 88611328][i]
    # DD
    # return [0, 289768960,509013504,667535360,761931264,793141760,762346496,668976640,512570368][i]
#     return [0, 117070336,
# 205512704,
# 271609856,
# 313274880,
# 327143424,
# 315484160,
# 279014400,
# 213555200][i]

def get_M_c(i):  # C模块的最终显存
    return [0, 25142272, 44344832, 59184640, 67560448, 69645312, 67535360, 59134976, 44684288][i]
    # DD
    # return [0, 143852980, 256133344, 337037548, 386565592, 404717476, 391493200, 346892764, 270916168][i]
#     return [0, 57374720,
# 102556672,
# 136465920,
# 159196160,
# 165908992,
# 161819648,
# 143800832,
# 112188928][i]


# 设置某些模块不使用 checkpoint（即保留最终显存）
x[1] = 0
x[2] = 0
x[3] = 0
x[4] = 0
x[5] = 1
x[6] = 1
x[7] = 1
x[8] = 1
x[9] = 1
x[10] =1
x[11] =1
x[12] =1
x[13] =1
x[14] =1
x[15] =1
x[16] =1
x[17] =0
x[18] =0
x[19] =0
x[20] =0
x[21] =1
x[22] =1
x[23] =1
x[24] =1
x[25] =1
x[26] =1
x[27] =1
x[28] =1
x[29] =1
x[30] =1
x[31] =1
x[32] =1
p = 5329920
# DD
# p=229034496
# p=95416832
back_peek = 314701824
forward_peek = 209345024
# back_peek = 2307730432
# forward_peek = 1235357696

# 计算显存峰值序列
K = compute_total_memory_peaks()
print("各时间段的显存峰值 K_t:", K[1:])

# 计算整体最大显存峰值
K_max = compute_overall_peak()
print("最大显存峰值 K_max:", K_max)





