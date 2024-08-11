import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
v = 500        # 注射用生理盐水体积 (ml)
k0 = 5         # 前2分钟的滴定速度 (mg/min)
k01 = 50 / 58  # 2-60分钟的滴定速度 (mg/min)
k02 = 40 / 120 # 60-180分钟的滴定速度 (mg/min)

# 数据
time = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                 80, 85, 90, 95, 100, 105, 120, 130, 140, 160, 180, 200, 210, 220, 240, 250])
conc = np.array([0.2547, 0.3139, 0.3343, 0.327, 0.3084, 0.2859, 0.2628, 0.2406,
                 0.2197, 0.2005, 0.1829, 0.1667, 0.152, 0.1386, 0.1263, 0.1151,
                 0.1049, 0.0957, 0.0872, 0.0795, 0.0724, 0.066, 0.05, 0.0416,
                 0.0345, 0.0238, 0.0165, 0.0114, 0.0094, 0.0078, 0.0054, 0.0045])

# 插值
f = interp1d(time, conc, kind='cubic')
time_interp = np.linspace(time.min(), time.max(), 1000)
conc_interp = f(time_interp)

# 定义模型函数
def model(t, k, m, m1, x120, Vd=22.84):
    result = np.zeros_like(t, dtype=float)
    
    mask_0_2 = t < 2
    mask_2_60 = (t >= 2) & (t < 60)
    mask_60_180 = (t >= 60) & (t < 180)
    mask_180_plus = t >= 180
    
    result[mask_0_2] = (1 - np.exp(-k*t[mask_0_2])) * k0 / (k*Vd)
    result[mask_2_60] = (k01 - m * np.exp(-k*(t[mask_2_60]-2))) / (k*Vd)
    result[mask_60_180] = (k02 - m1 * np.exp(-k*(t[mask_60_180]-60))) / (k*Vd)
    result[mask_180_plus] = x120 * np.exp(120*k) * np.exp(-k*(t[mask_180_plus]-180)) / Vd
    
    return result

# 使用插值后的数据进行拟合
popt, _ = curve_fit(model, time_interp, conc_interp, p0=[0.1, 20, 1, 1])

# 获取最佳拟合参数
k_fit, m_fit, m1_fit, x120_fit = popt

# 计算拟合曲线
time_points = np.linspace(0, 250, 1000)
conc_fit = model(time_points, k_fit, m_fit, m1_fit, x120_fit)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.scatter(time, conc, label='原始数据', color='red')
plt.plot(time_interp, conc_interp, label='插值数据', color='green', alpha=0.5)
plt.plot(time_points, conc_fit, label='拟合曲线', color='blue')
plt.title('血药浓度随时间变化')
plt.xlabel('时间 (min)')
plt.ylabel('血药浓度 (mg/L)')
plt.legend()
plt.grid(True)
plt.show()

# 输出拟合参数
print(f"拟合参数：k = {k_fit:.4f}, Vd = {22.84:.4f}, m = {m_fit:.4f}, m1 = {m1_fit:.4f}, x120 = {x120_fit:.4f}")