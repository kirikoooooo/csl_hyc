# from collections import Counter
#
# def count_labels_in_ts_file(file_path):
#     label_counter = Counter()
#
#     with open(file_path, encoding='utf-8') as file:
#         lines = file.readlines()
#         Start_reading_data = False
#
#         for line in lines:
#             if not Start_reading_data:
#                 if '@data' in line:
#                     Start_reading_data = True
#             else:
#                 temp = line.strip().split(':')
#                 label = temp[-1]  # 获取最后的类别标签
#                 label_counter[label] += 1  # 计数
#
#     # 检查是否有类别数量小于 10
#     if any(count < 10 for count in label_counter.values()):
#         return "no"
#
#     return label_counter
#
# # 示例调用
# file_path = "C:/Users/AW/Desktop/C-main/CSL-main/Multivariate_ts/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts"
# label_counts = count_labels_in_ts_file(file_path)
# print(label_counts)
import numpy as np
from utils import generate_binomial_mask
import tsaug
def countE():
    # cdist固定消耗
    Me = 75815424
    #batchsize
    b=4
    #column
    c=1345
    #length
    l=270
    #number
    s=13
    #shapelet_size
    k=[27,
54,
81,
108,
135,
162,
189,
216
]
    ans=[]
    for i in k:
        # ans.append(4*(c*s*i+b*c*l+b*c*(l-i+1)*i+b*c*(l-i+1)*s+b*c*s+b*c*(l-i+1)+b*(l-i+1)*s+b*s))
        # 欧氏距离
        ans.append(4 * ( b * c * (l - i + 1) * i ) +Me)
        # 余弦相似度
        # ans.append(4 * (
        #             2*c*s + c * s * i  + b * c * (l - i + 1) * i + b*s +b*(l-i+1)*s))
    return ans

print(countE())






















# data = {
#     14: 0.029173,
#     28: 0.0263736,
#     42: 0.0239874,
#     57: 0.0206732,
#     71: 0.0176732,
#     86: 0.01422,
#     100: 0.0110492,
#     115: 0.007536
# }
#
#
# def calculate_value(selected_keys):
#     if not all(key in data for key in selected_keys):
#         raise ValueError("选定的键中有不存在于数据中的")
#
#     sum_selected = sum(data[key] * 6 * 33 for key in selected_keys)
#     sum_remaining = sum(data[key] * 4 * 33 for key in data if key not in selected_keys)
#
#     result = sum_selected + sum_remaining
#     return result
#
#
# # 示例：选取 key = [14, 42]
# selected_keys = []
# # 14,28,42,57,71,86,100,115
# output = calculate_value(selected_keys)
# print(f"{output + 2.9423856:.6f}")

