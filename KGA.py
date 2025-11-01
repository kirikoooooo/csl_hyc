from sko.GA import GA
import numpy as np

def init_population_k_ones(size_pop, len_chrom, k):
    Chrom = np.zeros((size_pop, len_chrom), dtype=int)
    for i in range(size_pop):
        ones_idx = np.random.choice(len_chrom, size=k, replace=False)
        Chrom[i, ones_idx] = 1
    return Chrom

class KGA(GA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200,
                 prob_mut=0.001, lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, k=3):
        # 先设置 k
        self.k = k
        # 调用父类初始化
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub,
                         constraint_eq, constraint_ueq, precision, early_stop)
        # 确保 Chrom 被正确初始化
        self.Chrom = init_population_k_ones(size_pop, self.len_chrom, k)

    # 重写 crtbp 方法，确保每次创建种群都保持 k 个 1
    def crtbp(self):
        self.Chrom = init_population_k_ones(self.size_pop, self.len_chrom, self.k)
        return self.Chrom

# 自定义交叉算子函数 - 保持 k 个 1 的约束
def crossover_k_ones(algorithm, **kwargs):
    """
    自定义交叉算子，确保交叉后每个个体仍然保持 k 个 1
    """
    Chrom = algorithm.Chrom
    size_pop, len_chrom = Chrom.shape
    new_Chrom = np.zeros_like(Chrom)

    # 获取 k 值
    k = getattr(algorithm, 'k', 3)

    # 成对交叉
    for i in range(0, size_pop - 1, 2):
        if i + 1 < size_pop:
            # 执行标准两点交叉
            if len_chrom >= 2:  # 确保至少有2个基因
                idx1, idx2 = np.random.choice(len_chrom, 2, replace=False)
                start, end = min(idx1, idx2), max(idx1, idx2)

                # 交叉
                new_Chrom[i] = Chrom[i].copy()
                new_Chrom[i+1] = Chrom[i+1].copy()

                # 交换片段
                new_Chrom[i, start:end] = Chrom[i+1, start:end]
                new_Chrom[i+1, start:end] = Chrom[i, start:end]

                # 修复约束：确保每个个体都有 k 个 1
                for j in [i, i+1]:
                    current_sum = np.sum(new_Chrom[j])
                    if current_sum != k:
                        # 如果1的个数不对，重新生成
                        if len_chrom >= k:
                            ones_idx = np.random.choice(len_chrom, size=k, replace=False)
                            new_Chrom[j] = 0
                            new_Chrom[j, ones_idx] = 1
            else:
                # 如果基因长度小于2，直接复制
                new_Chrom[i] = init_population_k_ones(1, len_chrom, k)[0]
                if i + 1 < size_pop:
                    new_Chrom[i+1] = init_population_k_ones(1, len_chrom, k)[0]

    # 处理奇数个个体的情况
    if size_pop % 2 == 1:
        new_Chrom[-1] = init_population_k_ones(1, len_chrom, k)[0]

    return new_Chrom

# 自定义变异算子函数 - 保持 k 个 1 的约束
def mutation_swap_k_ones(algorithm, **kwargs):
    """
    自定义变异算子，通过交换保持 k 个 1 的约束
    """
    Chrom = algorithm.Chrom.copy()
    k = getattr(algorithm, 'k', 3)

    for i in range(Chrom.shape[0]):
        if np.random.rand() < algorithm.prob_mut:
            # 只交换不同值的位置（0和1）
            ones_idx = np.where(Chrom[i] == 1)[0]
            zeros_idx = np.where(Chrom[i] == 0)[0]

            if len(ones_idx) > 0 and len(zeros_idx) > 0:
                one_pos = np.random.choice(ones_idx)
                zero_pos = np.random.choice(zeros_idx)
                # 交换
                Chrom[i, one_pos], Chrom[i, zero_pos] = Chrom[i, zero_pos], Chrom[i, one_pos]

    return Chrom





    # # # 注册自定义变异算子
        # # ga.register(operator_name='mutation', operator=mutation_swap)
        # ga.register('crossover', crossover_k_ones)
        # ga.register('mutation', mutation_swap_k_ones)