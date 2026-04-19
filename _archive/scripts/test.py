import random
import time
import pulp

# 随机生成集合的函数
def generate_data(X_size, F_size):
    X = set(range(X_size))
    F = []

    # Step 1: 生成 S0，随机选 20 个点
    covered = set()
    S0 = set(random.sample(list(X), min(20, X_size)))
    F.append(S0)
    covered.update(S0)

    while len(covered) < X_size and len(F) < F_size:
        remaining = X - covered
        prev_covered = covered.copy()
        n = random.randint(1, 20)
        x = random.randint(1, min(n, len(remaining)))

        from_remaining = set(random.sample(list(remaining), x))
        from_covered = set(random.sample(list(prev_covered), n - x)) if prev_covered else set()

        new_set = from_remaining | from_covered
        F.append(new_set)
        covered.update(new_set)

    # Step 2: 生成剩下的集合
    while len(F) < F_size:
        size = random.randint(1, min(20, X_size))
        subset = set(random.sample(list(X), size))
        F.append(subset)

    return X, F

# 贪心算法
def greedy_set_cover(X, F):
    n = len(F)
    C = set()  # 已经选择的集合
    Us = set(X)  # 需要覆盖的元素
    res = []

    while Us:
        max_cover = 0
        best_set = -1
        for i in range(n):
            if i in C:
                continue  # 已选择的集合跳过
            current_cover = len(F[i] & Us)
            if current_cover > max_cover:
                max_cover = current_cover
                best_set = i

        if best_set == -1:
            break  # 如果没有找到合适的集合，退出

        res.append(best_set)
        C.add(best_set)
        Us -= F[best_set]

    return res

# LP 近似算法
def lp_rounding_set_cover(X, F):
    problem = pulp.LpProblem("SetCover", pulp.LpMinimize)
    vars = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(F))]

    # 目标函数：最小化选择的集合数
    problem += pulp.lpSum(vars)

    # 约束条件：确保每个元素都被至少一个集合覆盖
    for x in X:
        problem += pulp.lpSum(vars[i] for i, S in enumerate(F) if x in S) >= 1

    problem.solve()

    # 返回选中的集合
    selected_sets = [i for i in range(len(F)) if pulp.value(vars[i]) == 1]
    return selected_sets

# 测试函数
def test_set_cover(X_size, F_size):
    print(f"\nTesting with |X|={X_size}, |F|={F_size}")
    X, F = generate_data(X_size, F_size)

    print("Running Greedy Algorithm...")
    start = time.time()
    greedy_C = greedy_set_cover(X, F)
    print(f"Greedy Cover Size: {len(greedy_C)}, Time: {time.time() - start:.2f}s")

    print("Running LP Approximation...")
    start = time.time()
    lp_C = lp_rounding_set_cover(X, F)
    print(f"LP Cover Size: {len(lp_C)}, Time: {time.time() - start:.2f}s")

    # 验证结果是否有效
    def is_valid_cover(C):
        covered = set()
        for i in C:
            covered |= F[i]
        return covered == set(X)

    assert is_valid_cover(greedy_C), "Greedy did not produce valid cover"
    assert is_valid_cover(lp_C), "LP rounding did not produce valid cover"

    print("✅ Both solutions are valid covers.")

# 运行测试
for size in [5000]:
    test_set_cover(size, size)
