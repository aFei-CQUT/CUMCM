import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 定义新的模型函数
def model(t, k, V, a, b, c):
    D_rate = 5 * np.exp(-a*t) + 0.862 * np.exp(-b*t) + 0.333 * (1 - np.exp(-c*t))
    C = (D_rate / k) * (1 - np.exp(-k*t)) / V
    return C

# 设置参数边界
bounds = ([0, 1, 0, 0, 0], [1, 100, 2, 2, 2])

# 使用插值后的数据进行拟合
try:
    popt, pcov = curve_fit(model, time_interp, conc_interp, 
                           p0=[0.05, 25, 0.01, 0.01, 0.01], 
                           bounds=bounds, 
                           method='trf', 
                           maxfev=10000)

    # 获取最佳拟合参数
    k_fit, V_fit, a_fit, b_fit, c_fit = popt

    # 计算拟合曲线
    time_points = np.linspace(0, 250, 1000)
    conc_fit = model(time_points, k_fit, V_fit, a_fit, b_fit, c_fit)

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
    print(f"拟合参数：k = {k_fit:.4f}, V = {V_fit:.4f}, a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}")

    # 计算半衰期
    t_half = np.log(2) / k_fit
    print(f"半衰期：{t_half:.2f} min")

    # 计算AUC
    AUC = np.trapz(conc_fit, time_points)
    print(f"AUC：{AUC:.4f} mg·min/L")

except RuntimeError as e:
    print("拟合失败:", str(e))
