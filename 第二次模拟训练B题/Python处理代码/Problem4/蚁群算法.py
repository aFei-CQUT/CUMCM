import numpy as np
import matplotlib.pyplot as plt

# 常量定义
weight = 56  # kg，体重
dose_per_kg = 0.9  # mg/kg，单位体重的剂量
Vd = 22.84  # L，分布容积
t_half = 5  # min，半衰期
k = np.log(2) / t_half  # 消除率常数 (min^-1)
total_dose = min(weight * dose_per_kg, 90)  # 最大总剂量为90 mg
dose1 = 0.1 * total_dose  # 总剂量的10%
dose2 = 0.9 * total_dose  # 总剂量的90%

# 定义浓度函数
def concentration(t, t0, t1):
    if t < t0:
        return (dose1 / Vd) * np.exp(-k * t)
    else:
        k0 = dose2 / (t1 - t0)
        return ((dose1 / Vd) * np.exp(-k * t) + 
                (k0 / (Vd * k)) * (1 - np.exp(-k * (t - t0))))

# 蚁群算法
def ant_colony_optimization(num_ants=30, num_iterations=100):
    pheromone = np.ones((60, 180)) * 1e-3  # 初始化信息素
    best_solution = None
    best_fitness = float('inf')

    for _ in range(num_iterations):
        solutions = []
        fitness_values = []

        for _ in range(num_ants):
            t0 = np.random.randint(0, 61)  # t0 在 [0, 60]
            t1 = np.random.randint(60, 181)  # t1 在 [60, 180]

            # 计算适应度
            if t1 > t0:
                times = np.linspace(0, t1, 1000)
                concentrations = [concentration(t, t0, t1) for t in times]
                min_conc = min(concentrations)
                max_conc = max(concentrations)
                
                if min_conc >= 0.18 and max_conc <= 0.24:
                    fitness = t1  # 以t1为目标进行优化
                else:
                    fitness = float('inf')  # 不符合约束条件
            else:
                fitness = float('inf')

            solutions.append((t0, t1))
            fitness_values.append(fitness)

            # 更新最佳解
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = (t0, t1)

        # 信息素更新
        pheromone *= 0.95  # 信息素挥发
        for i in range(num_ants):
            if fitness_values[i] < float('inf'):
                pheromone[solutions[i][0], solutions[i][1]] += 1.0 / fitness_values[i]  # 增加信息素

    return best_solution

# 运行蚁群算法
best_solution = ant_colony_optimization()
t0_opt, t1_opt = best_solution
infusion_rate = dose2 / (t1_opt - t0_opt)

print(f"最优间隔时间 t0 = {t0_opt:.2f} min")
print(f"最优滴注时间 t1 = {t1_opt:.2f} min")
print(f"滴注速度 = {infusion_rate:.4f} mg/min")

# 验证结果
times = np.linspace(0, t1_opt, 1000)
concentrations = [concentration(t, t0_opt, t1_opt) for t in times]
print(f"最小浓度: {min(concentrations):.4f} mg/L")
print(f"最大浓度: {max(concentrations):.4f} mg/L")

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(times, concentrations, 'b-', label='血药浓度')
plt.axhline(y=0.18, color='r', linestyle='--', label='最小治疗浓度')
plt.axhline(y=0.24, color='g', linestyle='--', label='最大治疗浓度')
plt.axvline(x=t0_opt, color='m', linestyle=':', label='开始滴注时间')

plt.xlabel('时间 (分钟)')
plt.ylabel('血药浓度 (mg/L)')
plt.title('血药浓度随时间的变化')
plt.legend()
plt.grid(True)
plt.show()