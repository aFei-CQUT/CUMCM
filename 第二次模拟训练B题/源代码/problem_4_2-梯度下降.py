import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    return x[1]  # 最小化t1

# 定义约束条件
def constraint(x):
    t0, t1 = x
    times = np.linspace(0, t1, 1000)
    concentrations = [concentration(t, t0, t1) for t in times]
    return np.min(concentrations) - 0.18, 0.24 - np.max(concentrations)

# 绘图函数
def plot_concentration(t0, t1):
    times = np.linspace(0, t1, 1000)
    concentrations = [concentration(t, t0, t1) for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, concentrations, 'b-', label='血药浓度')
    plt.axhline(y=0.18, color='r', linestyle='--', label='最小治疗浓度')
    plt.axhline(y=0.24, color='g', linestyle='--', label='最大治疗浓度')
    plt.axvline(x=t0, color='m', linestyle=':', label='开始滴注时间')
    
    plt.xlabel('时间 (分钟)')
    plt.ylabel('血药浓度 (mg/L)')
    plt.title('血药浓度随时间的变化')
    plt.legend()
    plt.grid(True)
    plt.show()

# 初始猜测
x0 = [5, 120]  # 初始猜测t0=30min, t1=120min

# 约束条件
cons = {'type': 'ineq', 'fun': constraint}
bounds = [(0, None), (60, None)]  # t0 >= 0, t1 >= 60

# 求解优化问题trust-constr算法表现更好
res = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons)

# 输出结果
if res.success:
    t0_opt, t1_opt = res.x
    print(f"最优间隔时间 t0 = {t0_opt:.2f} min")
    print(f"最优滴注时间 t1 = {t1_opt:.2f} min")
    print(f"滴注速度 = {dose2 / (t1_opt - t0_opt):.4f} mg/min")

    # 验证结果
    times = np.linspace(0, t1_opt, 1000)
    concentrations = [concentration(t, t0_opt, t1_opt) for t in times]
    print(f"最小浓度: {min(concentrations):.4f} mg/L")
    print(f"最大浓度: {max(concentrations):.4f} mg/L")

    # 绘制图形
    plot_concentration(t0_opt, t1_opt)
else:
    print("优化失败，未找到满足条件的解。")
    print("优化器状态:", res.message)

# 如果优化失败，我们可以尝试不同的初始值
if not res.success:
    print("\n尝试不同的初始值：")
    for t0_init in [10, 20, 40, 50]:
        for t1_init in [90, 150, 180, 210]:
            x0 = [t0_init, t1_init]
            res = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons)
            if res.success:
                t0_opt, t1_opt = res.x
                print("\n找到可行解：")
                print(f"最优间隔时间 t0 = {t0_opt:.2f} min")
                print(f"最优滴注时间 t1 = {t1_opt:.2f} min")
                print(f"滴注速度 = {dose2 / (t1_opt - t0_opt):.4f} mg/min")

                # 验证结果
                times = np.linspace(0, t1_opt, 1000)
                concentrations = [concentration(t, t0_opt, t1_opt) for t in times]
                print(f"最小浓度: {min(concentrations):.4f} mg/L")
                print(f"最大浓度: {max(concentrations):.4f} mg/L")

                # 绘制图形
                plot_concentration(t0_opt, t1_opt)
                break
        if res.success:
            break
    else:
        print("尝试多个初始值后仍未找到满足条件的解。")
