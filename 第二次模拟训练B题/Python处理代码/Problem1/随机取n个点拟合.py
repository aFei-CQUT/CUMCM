import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import sympy as sp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用sympy求解微分方程
t, k, V = sp.symbols('t k V')
C = sp.Function('C')(t)
eq = sp.Eq(C.diff(t), -k*C)
general_solution = sp.dsolve(eq, C)
print("通解:", general_solution)

# 应用初始条件 C(0) = D/V (D = 50, F = 1)
D = 50  # 给药剂量
F = 1   # 生物利用度(静脉注射)
C0 = D/V
particular_solution = general_solution.rhs.subs(general_solution.rhs.subs(t, 0), C0)
print("特解:", particular_solution)

# 原始数据
time = np.array([i for i in range(1, 21)])
concentration = np.array([
    1.3158, 1.1405, 0.9885, 0.8568, 0.7426, 0.6437, 0.5579, 
    0.4836, 0.4191, 0.3633, 0.3149, 0.2729, 0.2366, 0.205, 
    0.1777, 0.154, 0.1335, 0.1157, 0.1003, 0.0754
])

# 随机选择fit_size个点进行拟合
fit_size = 6
fit_indices = np.random.choice(len(time), size=fit_size, replace=False)
time_fit = time[fit_indices]
concentration_fit = concentration[fit_indices]

# 定义一阶动力学模型函数
def first_order_model(t, V, k):
    return (D / V) * np.exp(-k * t)

# 使用curve_fit进行拟合
popt, pcov = curve_fit(first_order_model, time_fit, concentration_fit, p0=[30, 0.1])

# 获取拟合参数
V_fit, k_fit = popt

# 计算半衰期
t_half = np.log(2) / k_fit

# 生成拟合曲线的点,包括0时刻
t_fit = np.linspace(0, 20, 100)
c_fit = first_order_model(t_fit, V_fit, k_fit)

# 计算AUC
AUC = np.trapz(c_fit, t_fit)

# 计算残差（对所有点）
residuals_all = concentration - first_order_model(time, V_fit, k_fit)

# 创建四个子图
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# 1. 原图
axs[0, 0].scatter(time, concentration, label='所有数据')
axs[0, 0].scatter(time_fit, concentration_fit, color='red', label='用于拟合的数据')
axs[0, 0].plot(t_fit, c_fit, 'g-', label='拟合曲线')
axs[0, 0].set_xlabel('时间 (小时)')
axs[0, 0].set_ylabel('浓度 (mg/L)')
axs[0, 0].set_title('药物浓度随时间变化')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. 对数转换
axs[0, 1].scatter(time, np.log(concentration), label='所有数据')
axs[0, 1].scatter(time_fit, np.log(concentration_fit), color='red', label='用于拟合的数据')
axs[0, 1].plot(t_fit, np.log(c_fit), 'g-', label='拟合曲线')
axs[0, 1].set_xlabel('时间 (小时)')
axs[0, 1].set_ylabel('ln(浓度)')
axs[0, 1].set_title('药物浓度对数随时间变化')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. 残差图
axs[1, 0].scatter(time, residuals_all)
axs[1, 0].axhline(y=0, color='r', linestyle='--')
axs[1, 0].set_xlabel('时间 (小时)')
axs[1, 0].set_ylabel('残差')
axs[1, 0].set_title('残差图')
axs[1, 0].grid(True)

# 4. QQ图
stats.probplot(residuals_all, dist="norm", plot=axs[1, 1])
axs[1, 1].set_title("Q-Q 图")

plt.tight_layout()
plt.show()

# 打印结果
print("拟合参数:")
print(f"D (给药剂量) = {D:.4f} mg (固定值)")
print(f"F (生物利用度) = {F:.4f} (固定值)")
print(f"V (分布容积) = {V_fit:.4f} L")
print(f"k (消除速率常数) = {k_fit:.4f} h^-1")
print(f"C0 (0时刻浓度) = {D / V_fit:.4f} mg/L")
print(f"半衰期 = {t_half:.2f} 小时")
print(f"AUC = {AUC:.2f} mg·h/L")

# 统计量分析（使用所有点）
r_squared = 1 - (np.sum(residuals_all**2) / np.sum((concentration - np.mean(concentration))**2))
print(f"R-squared: {r_squared:.4f}")

# 进行Shapiro-Wilk正态性检验（使用所有点）
_, p_value = stats.shapiro(residuals_all)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")