import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from sklearn.model_selection import LeaveOneOut
import sympy as sp
from scipy import stats
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 符号计算部分
t = sp.Symbol('t')
x = sp.Function('x')(t)
k1, k2, F, D, V, C0 = sp.symbols('k1 k2 F D V C0')

eq = sp.Eq(sp.diff(x, t), k1*(F*D - x) - k2*x)
general_solution = sp.dsolve(eq, x)

print("微分方程:")
sp.pprint(eq)
print("\n通解 x(t):")
sp.pprint(general_solution.rhs)

# 应用初始条件 x(0) = C0*V
C1 = sp.symbols('C1')
initial_condition = general_solution.rhs.subs(t, 0)
C1_value = sp.solve(sp.Eq(initial_condition, C0*V), C1)[0]

print("\n应用初始条件 x(0) = C0*V 后的 C1 值:")
sp.pprint(C1_value)

# 代入 C1 值得到特解
particular_solution = general_solution.rhs.subs(C1, C1_value)
simplified_solution = sp.simplify(particular_solution)

print("\n特解 x(t):")
sp.pprint(simplified_solution)

# 计算浓度 c(t)
c = simplified_solution / V
c_simplified = sp.simplify(c.expand())

print("\n浓度 c(t) 的表达式:")
sp.pprint(c_simplified)

# 计算峰值时间 t*
t_star = sp.log((k2*(F*D + V*C0))/(k1*F*D)) / (k2 - k1)
print("\n峰值时间 t*:")
sp.pprint(t_star)

# 数值拟合部分
# 数据
time = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                 80, 85, 90, 95, 100, 105, 120, 130, 140, 160, 180, 200, 210, 220, 240, 250])
concentration = np.array([0.2547, 0.3139, 0.3343, 0.327, 0.3084, 0.2859, 0.2628, 0.2406,
                          0.2197, 0.2005, 0.1829, 0.1667, 0.152, 0.1386, 0.1263, 0.1151,
                          0.1049, 0.0957, 0.0872, 0.0795, 0.0724, 0.066, 0.05, 0.0416,
                          0.0345, 0.0238, 0.0165, 0.0114, 0.0094, 0.0078, 0.0054, 0.0045])

# 定义新的模型函数
def model(t, k1, k2, V, C0):
    F = 1
    D = 50
    return C0 * np.exp(-k2 * t) + (k1 * F * D) / (V * (k1 - k2)) * (np.exp(-k2 * t) - np.exp(-k1 * t))

# 拟合曲线
popt, pcov = curve_fit(model, time, concentration, p0=[0.1, 0.01, 30, 0.3])
k1, k2, V, C0 = popt

# 计算预测值
predicted = model(time, k1, k2, V, C0)

# 计算理论AUC
def auc_func(t):
    return model(t, k1, k2, V, C0)

theoretical_auc, _ = quad(auc_func, 0, np.inf)

# 计算数值积分AUC (拟合曲线)
numerical_auc, _ = quad(auc_func, 0, time[-1])

# 计算数值积分AUC (三次样条插值)
cs = CubicSpline(time, concentration)
numerical_auc_spline, _ = quad(cs, time[0], time[-1])

# 计算半衰期
half_life = np.log(2) / k2

# 计算峰值时间
t_max = np.log((k2*(50 + V*C0))/(k1*50)) / (k2 - k1)

# 留一法交叉验证
loo = LeaveOneOut()
errors = []
for train_index, test_index in loo.split(time):
    time_train, time_test = time[train_index], time[test_index]
    conc_train, conc_test = concentration[train_index], concentration[test_index]
    popt, _ = curve_fit(model, time_train, conc_train, p0=[0.1, 0.01, 30, 0.3])
    pred = model(time_test, *popt)
    errors.append((pred - conc_test)**2)

mean_error = np.mean(errors)

# 计算残差
residuals = concentration - predicted

# 计算R方
ss_res = np.sum(residuals**2)
ss_tot = np.sum((concentration - np.mean(concentration))**2)
r_squared = 1 - (ss_res / ss_tot)

# 计算均方根误差（RMSE）
rmse = np.sqrt(np.mean(residuals**2))

# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(residuals))

# 进行Shapiro-Wilk正态性检验
_, p_value = stats.shapiro(residuals)

# 输出结果
print(f"\n拟合后的方程表达式: c(t) = {C0:.4f} * exp(-{k2:.4f} * t) + {k1*50/(V*(k1-k2)):.4f} * (exp(-{k2:.4f} * t) - exp(-{k1:.4f} * t))")
print(f"拟合参数: k1 = {k1:.4f}, k2 = {k2:.4f}, V = {V:.4f}, C0 = {C0:.4f} (F=1, D=50)")

# 创建统计量数据
stats_data = {
    '统计量': [
        '理论AUC',
        '数值积分AUC (拟合曲线)',
        '数值积分AUC (三次样条插值)',
        '半衰期',
        '峰值时间',
        '留一法交叉验证平均误差',
        'R方',
        'RMSE',
        'MAE',
        'Shapiro-Wilk检验p值'
    ],
    '值': [
        theoretical_auc,
        numerical_auc,
        numerical_auc_spline,
        half_life,
        t_max,
        mean_error,
        r_squared,
        rmse,
        mae,
        p_value
    ]
}

# 创建DataFrame
df_stats = pd.DataFrame(stats_data)

# 设置'统计量'列为索引
df_stats.set_index('统计量', inplace=True)

# 格式化'值'列
df_stats['值'] = df_stats['值'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))

# 显示表格
print("\n统计量表格:")
print(df_stats.to_string())

# 绘制拟合曲线和原始数据点
plt.figure(figsize=(12, 9))
plt.subplot(221)
plt.scatter(time, concentration, label='原始数据')
t_smooth = np.linspace(0, time[-1], 1000)
plt.plot(t_smooth, model(t_smooth, *popt), 'r-', label='拟合曲线')
plt.xlabel('时间 (min)')
plt.ylabel('浓度 (mg/mL)')
plt.title('血药浓度随时间变化曲线')
plt.legend()
plt.grid(True)

# 浓度坐标轴对数转换
plt.subplot(222)
plt.scatter(time, np.log(concentration), label='原始数据')
plt.plot(t_smooth, np.log(model(t_smooth, *popt)), 'r-', label='拟合曲线')
plt.xlabel('时间 (min)')
plt.ylabel('ln(浓度) (mg/mL)')
plt.title('对数转换后的浓度-时间曲线')
plt.legend()
plt.grid(True)

# 残差图
plt.subplot(223)
plt.scatter(time, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('时间 (min)')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True)

# QQ图
plt.subplot(224)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("残差QQ图")

plt.tight_layout()
plt.show()
