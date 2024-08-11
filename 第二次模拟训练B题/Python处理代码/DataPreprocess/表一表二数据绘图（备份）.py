import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson, trapezoid

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 静脉注射数据
iv_inject_time = [i for i in range(1, 21)]
iv_inject_concentration = [
    1.3158, 1.1405, 0.9885, 0.8568, 0.7426, 0.6437, 0.5579, 
    0.4836, 0.4191, 0.3633, 0.3149, 0.2729, 0.2366, 0.205, 
    0.1777, 0.154, 0.1335, 0.1157, 0.1003, 0.0754
]

# 静脉滴注数据
iv_drip_time = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                80, 85, 90, 95, 100, 105, 120, 130, 140, 160, 180, 200, 210, 220, 240, 250]
iv_drip_concentration = [
    0.2547, 0.3139, 0.3343, 0.327, 0.3084, 0.2859, 0.2628, 0.2406,
    0.2197, 0.2005, 0.1829, 0.1667, 0.152, 0.1386, 0.1263, 0.1151,
    0.1049, 0.0957, 0.0872, 0.0795, 0.0724, 0.066, 0.05, 0.0416,
    0.0345, 0.0238, 0.0165, 0.0114, 0.0094, 0.0078, 0.0054, 0.0045
]

# 插值处理
iv_inject_interp = interp1d(iv_inject_time, iv_inject_concentration, kind='cubic')
iv_drip_interp = interp1d(iv_drip_time, iv_drip_concentration, kind='cubic')

# 创建细分时间点用于插值
fine_iv_inject_time = np.linspace(1, 20, 100)
fine_iv_drip_time = np.linspace(1, 250, 100)

# 计算插值结果
fine_iv_inject_concentration = iv_inject_interp(fine_iv_inject_time)
fine_iv_drip_concentration = iv_drip_interp(fine_iv_drip_time)

# 创建图形：使用subplot绘制静脉注射和静脉滴注浓度曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# 静脉注射图形
ax1.plot(fine_iv_inject_time, fine_iv_inject_concentration, label='静脉注射 (插值)', color='red')
ax1.scatter(iv_inject_time, iv_inject_concentration, color='blue', label='静脉注射 (原始数据)')
ax1.set_title('静脉注射药物浓度曲线')
ax1.set_xlabel('时间 (min)')
ax1.set_ylabel('血药浓度 (mg/L)')
ax1.legend()
ax1.grid(True, which='both')
ax1.minorticks_on()
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)

# 静脉滴注图形
ax2.plot(fine_iv_drip_time, fine_iv_drip_concentration, label='静脉滴注 (插值)', color='orange')
ax2.scatter(iv_drip_time, iv_drip_concentration, color='green', label='静脉滴注 (原始数据)')
ax2.set_title('静脉滴注药物浓度曲线')
ax2.set_xlabel('时间 (min)')
ax2.set_ylabel('血药浓度 (mg/L)')
ax2.legend()
ax2.grid(True, which='both')
ax2.minorticks_on()
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)

plt.tight_layout()
plt.show()

# 数值积分
iv_inject_area_simps = simpson(y=fine_iv_inject_concentration, x=fine_iv_inject_time)
iv_drip_area_simps = simpson(y=fine_iv_drip_concentration, x=fine_iv_drip_time)

iv_inject_area_trapz = trapezoid(y=fine_iv_inject_concentration, x=fine_iv_inject_time)
iv_drip_area_trapz = trapezoid(y=fine_iv_drip_concentration, x=fine_iv_drip_time)

print("静脉注射药物浓度曲线面积（辛普森法）: ", iv_inject_area_simps)
print("静脉滴注药物浓度曲线面积（辛普森法）: ", iv_drip_area_simps)

print("静脉注射药物浓度曲线面积（梯形法）: ", iv_inject_area_trapz)
print("静脉滴注药物浓度曲线面积（梯形法）: ", iv_drip_area_trapz)