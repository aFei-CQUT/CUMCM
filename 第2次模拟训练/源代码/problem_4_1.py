import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数
W = 56  # 体重 (kg)
y = 0.9  # 药物剂量 (mg/kg)
Vd = 22.84  # 表观分布容积 (L)
t_half = 5  # 药物半衰期 (min)
k = np.log(2) / t_half  # 消除速率常数

# 问题1: 静脉滴注与静脉注射同时给药
Cs = 0.19           # 目标稳态浓度 (mg/L)
X0 = Cs * Vd        # 静脉注射给药剂量
k0 = Cs * Vd * k    # 静脉滴注速度

print("\n问题1:")
print(f"静脉注射的消除速率常数为：{k}")
print(f"静脉注射给药剂量: {X0:.2f} mg")
print(f"静脉滴注速度: {k0:.2f} mg/min")

# 绘图
t = np.linspace(0, 120, 1000)  # 时间范围0-120分钟,1000个点
C = (X0 * np.exp(-k*t) + k0/k * (1 - np.exp(-k*t))) / Vd
# C = (X0 * np.exp(-k*t) + k0/k * (1 - np.exp(-k*t))) / Vd + k1*F*D*(np.exp(-k1*t))

plt.figure(figsize=(10, 6))
plt.plot(t, C, label='血药浓度')
plt.axhline(y=Cs, color='r', linestyle='--', label='目标浓度')
plt.xlabel('时间 (min)')
plt.ylabel('血药浓度 (mg/L)')
plt.title('静脉滴注与静脉注射同时给药的血药浓度-时间曲线')
plt.legend()
plt.grid(True)
plt.show()
