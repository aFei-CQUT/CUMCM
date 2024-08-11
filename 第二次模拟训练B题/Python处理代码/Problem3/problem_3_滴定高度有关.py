import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义滴定速率函数
def Q(t, k1, k2, k3):
    A = 9  # 固定A为9平方厘米
    h0 = 1.5  # 固定h0为1.5米
    t1, t2 = 2, 60  # 第一阶段和第二阶段的结束时间
    result = np.zeros_like(t)
    
    mask1 = t <= t1
    mask2 = (t > t1) & (t <= t2)
    mask3 = t > t2
    
    result[mask1] = k1 * (np.sqrt(h0) - k1*t[mask1] / (2*A))
    
    h1 = (np.sqrt(h0) - k1*t1 / (2*A))**2
    result[mask2] = k2 * (np.sqrt(h1) - k2*(t[mask2]-t1) / (2*A))
    
    h2 = (np.sqrt(h1) - k2*(t2-t1) / (2*A))**2
    result[mask3] = k3 * (np.sqrt(h2) - k3*(t[mask3]-t2) / (2*A))
    
    return result

# 定义药物浓度模型
def model(t, k, k1, k2, k3):
    V = 22.84  # 固定Vd为22.84
    D_rate = Q(t, k1, k2, k3)
    C = (D_rate / k) * (1 - np.exp(-k*t)) / V
    return C

# 数据
time = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                 80, 85, 90, 95, 100, 105, 120, 130, 140, 160, 180, 200, 210, 220, 240, 250])
conc = np.array([0.2547, 0.3139, 0.3343, 0.327, 0.3084, 0.2859, 0.2628, 0.2406,
                 0.2197, 0.2005, 0.1829, 0.1667, 0.152, 0.1386, 0.1263, 0.1151,
                 0.1049, 0.0957, 0.0872, 0.0795, 0.0724, 0.066, 0.05, 0.0416,
                 0.0345, 0.0238, 0.0165, 0.0114, 0.0094, 0.0078, 0.0054, 0.0045])

# 设置参数边界
bounds = ([0, 0, 0, 0], [1, 10, 10, 10])

# 进行拟合
try:
    popt, pcov = curve_fit(model, time, conc, 
                           p0=[0.05, 1, 0.5, 0.1], 
                           bounds=bounds, 
                           method='trf', 
                           maxfev=10000)

    # 获取最佳拟合参数
    k_fit, k1_fit, k2_fit, k3_fit = popt

    # 计算拟合曲线
    time_points = np.linspace(0, 250, 1000)
    conc_fit = model(time_points, k_fit, k1_fit, k2_fit, k3_fit)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.scatter(time, conc, label='原始数据', color='red')
    plt.plot(time_points, conc_fit, label='拟合曲线', color='blue')
    plt.title('血药浓度随时间变化')
    plt.xlabel('时间 (min)')
    plt.ylabel('血药浓度 (mg/L)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 输出拟合参数
    print(f"拟合参数：k = {k_fit:.4f}, V = 22.8400 (固定值)")
    print(f"k1 = {k1_fit:.4f}, k2 = {k2_fit:.4f}, k3 = {k3_fit:.4f}")
    print(f"A = 9.0000, h0 = 1.5000 (固定值)")

    # 计算半衰期
    t_half = np.log(2) / k_fit
    print(f"半衰期：{t_half:.2f} min")

    # 计算AUC
    AUC = np.trapz(conc_fit, time_points)
    print(f"AUC：{AUC:.4f} mg·min/L")

except RuntimeError as e:
    print("拟合失败:", str(e))
