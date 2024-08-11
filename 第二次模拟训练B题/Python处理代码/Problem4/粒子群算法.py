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

# 定义适应度函数
def fitness(x):
    t0, t1 = x
    if t1 <= t0 or t0 < 0 or t1 < 60:
        return float('inf')  # 不符合约束条件
    times = np.linspace(0, t1, 1000)
    concentrations = [concentration(t, t0, t1) for t in times]
    min_conc = min(concentrations)
    max_conc = max(concentrations)
    
    # 目标是最小化t1，同时确保浓度在范围内
    if min_conc < 0.18 or max_conc > 0.24:
        return float('inf')  # 不符合浓度约束
    return t1  # 以t1为目标进行优化

# 粒子群优化算法
def particle_swarm_optimization(num_particles=30, max_iter=100):
    # 初始化粒子位置和速度
    particles = np.random.rand(num_particles, 2)
    particles[:, 0] *= 60  # t0 在 [0, 60]
    particles[:, 1] *= 120 + 60  # t1 在 [60, 180]
    
    velocities = np.random.rand(num_particles, 2) * 0.1  # 初始速度
    pbest = particles.copy()  # 个体最佳位置
    pbest_fitness = np.array([fitness(p) for p in pbest])  # 个体最佳适应度
    gbest = pbest[np.argmin(pbest_fitness)]  # 全局最佳位置
    
    for _ in range(max_iter):
        for i in range(num_particles):
            # 更新速度
            r1, r2 = np.random.rand(2)
            w = 0.5  # 惯性权重
            c1 = 1.5  # 个体学习因子
            c2 = 1.5  # 社会学习因子
            
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            
            # 更新位置
            particles[i] += velocities[i]
            # 确保位置在边界内
            particles[i, 0] = np.clip(particles[i, 0], 0, 60)
            particles[i, 1] = np.clip(particles[i, 1], 60, 180)
            
            # 评估适应度
            current_fitness = fitness(particles[i])
            if current_fitness < pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = current_fitness
                if current_fitness < fitness(gbest):
                    gbest = particles[i]
    
    return gbest

# 运行粒子群优化算法
best_solution = particle_swarm_optimization()
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