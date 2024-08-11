import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# 定义目标函数
def objective(x):
    t0, t1 = x
    if t1 <= t0 or t0 < 0 or t1 < 60:
        return 1e10  # 不符合约束条件
    times = np.linspace(0, t1, 1000)
    concentrations = [concentration(t, t0, t1) for t in times]
    min_conc = min(concentrations)
    max_conc = max(concentrations)
    
    if min_conc < 0.18 or max_conc > 0.24:
        return 1e10  # 不符合浓度约束
    return t1  # 以t1为目标进行优化

# 粒子群优化算法
def particle_swarm_optimization(num_particles=30, max_iter=100, objective_func=objective):
    particles = np.random.rand(num_particles, 2)
    particles[:, 0] *= 60  # t0 在 [0, 60]
    particles[:, 1] *= 120 + 60  # t1 在 [60, 180]
    
    velocities = np.random.rand(num_particles, 2) * 0.1
    pbest = particles.copy()
    pbest_fitness = np.array([objective_func(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_fitness)]
    
    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            w, c1, c2 = 0.5, 1.5, 1.5
            
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            
            particles[i] += velocities[i]
            particles[i, 0] = np.clip(particles[i, 0], 0, 60)
            particles[i, 1] = np.clip(particles[i, 1], 60, 180)
            
            current_fitness = objective_func(particles[i])
            if current_fitness < pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = current_fitness
                if current_fitness < objective_func(gbest):
                    gbest = particles[i]
    
    return gbest

# 蚁群优化算法
def ant_colony_optimization(num_ants=30, num_iterations=100, objective_func=objective):
    pheromone = np.ones((60, 180)) * 1e-3
    best_solution = None
    best_fitness = float('inf')

    for _ in range(num_iterations):
        solutions = []
        fitness_values = []

        for _ in range(num_ants):
            t0 = np.random.randint(0, 60)  # t0 在 [0, 60)
            t1 = np.random.randint(60, 181)  # t1 在 [60, 180]

            fitness = objective_func([t0, t1])
            solutions.append((t0, t1))
            fitness_values.append(fitness)

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = (t0, t1)

        pheromone *= 0.95
        for i in range(num_ants):
            if fitness_values[i] < float('inf'):
                t0_index = solutions[i][0]  # t0
                t1_index = solutions[i][1]  # t1
                # 确保 t1_index 在有效范围内
                if t1_index < 180:  # 只在有效范围内更新信息素
                    pheromone[t0_index, t1_index] += 1.0 / fitness_values[i]

    return best_solution

# 主函数
def main():
    # 运行三种优化算法
    pso_solution = particle_swarm_optimization()
    sa_solution = dual_annealing(objective, bounds=[(0, 60), (60, 180)], maxiter=1000).x
    aco_solution = ant_colony_optimization()

    print("PSO Solution:", pso_solution)
    print("SA Solution:", sa_solution)
    print("ACO Solution:", aco_solution)

    # 生成更多数据点用于机器学习
    num_samples = 1000
    X = np.random.rand(num_samples, 2)
    X[:, 0] *= 60  # t0 在 [0, 60]
    X[:, 1] = X[:, 1] * 120 + 60  # t1 在 [60, 180]
    y = np.array([objective(x) for x in X])

    # 训练随机森林模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # 使用模型预测最优解
    def rf_objective(x):
        return rf_model.predict([x])[0]

    # 使用模型预测的目标函数重新运行优化算法
    rf_pso_solution = particle_swarm_optimization(objective_func=rf_objective)
    rf_sa_solution = dual_annealing(rf_objective, bounds=[(0, 60), (60, 180)], maxiter=1000).x
    rf_aco_solution = ant_colony_optimization(objective_func=rf_objective)

    print("RF-PSO Solution:", rf_pso_solution)
    print("RF-SA Solution:", rf_sa_solution)
    print("RF-ACO Solution:", rf_aco_solution)

    # 比较所有结果,选择最佳解决方案
    all_solutions = [pso_solution, sa_solution, aco_solution, 
                     rf_pso_solution, rf_sa_solution, rf_aco_solution]
    final_best_solution = min(all_solutions, key=lambda x: objective(x))

    t0_opt, t1_opt = final_best_solution
    infusion_rate = dose2 / (t1_opt - t0_opt)

    print("\n最终最优解:")
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

if __name__ == "__main__":
    main()
