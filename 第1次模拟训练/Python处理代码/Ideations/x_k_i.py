import sympy as sp

# 初始种群数量分布向量
x0 = sp.Matrix([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

# 出生率向量 a
a = sp.Matrix([0, 0, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]).T  # 转置为行向量

# 存活率向量 b
b = sp.Matrix([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]).T  # 转置为行向量

# 定义符号变量 r, 出售刚出生的小母牛的比例
r = sp.symbols('r')

# 构建莱斯利空矩阵 L
n = len(x0)
L = sp.zeros(n)

# 考虑出售刚比率 r 刚出生的小母牛后，莱斯利矩阵 L 第一行赋值
L[0, :] = (1 - r) * a

# 莱斯利矩阵 L 每个年龄段存活率赋值
for i in range(1, n):
    L[i, i-1] = b[i-1]

# 计算种群数量分布向量并返回每年的结果
def compute_population_distribution(x0, L, k):
    results = [x0.copy()]  # 初始化结果列表，包含初始种群
    x = x0
    for _ in range(k):
        x = L * x
        results.append(x.copy())  # 记录每年的种群分布
    return results

# 计算每年的总种群数量
def compute_total_population(distributions):
    return [sum(dist) for dist in distributions]

# 设定计算的时间步数 k
k = 5

# 计算每一年的种群数量分布向量
population_distributions = compute_population_distribution(x0, L, k)

# 计算总种群数量
total_population = compute_total_population(population_distributions)
