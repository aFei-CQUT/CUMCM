import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import stats

# 原始数据
time = np.array([0] + [i for i in range(1, 21)])  # 添加0时刻
concentration = np.array([50/30] + [  # 添加0时刻的理论浓度
    1.3158, 1.1405, 0.9885, 0.8568, 0.7426, 0.6437, 0.5579, 
    0.4836, 0.4191, 0.3633, 0.3149, 0.2729, 0.2366, 0.205, 
    0.1777, 0.154, 0.1335, 0.1157, 0.1003, 0.0754
])

D = 50  # 给药剂量

# 选择用于拟合的点
fit_start = 10
time_fit = time[fit_start:]
concentration_fit = concentration[fit_start:]

def model(t, V, k):
    return (D / V) * np.exp(-k * t)

def log_likelihood(theta, t, y, yerr):
    V, k = theta
    model_pred = model(t, V, k)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model_pred)**2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    V, k = theta
    if 10 < V < 100 and 0.01 < k < 1:
        return 0.0
    return -np.inf

def log_probability(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr)

# 初始猜测
initial_guess = [30, 0.1]
ndim = len(initial_guess)
nwalkers = 32

# 初始化 walkers
pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

# 设置 emcee sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(time_fit, concentration_fit, concentration_fit*0.1))

# 运行 MCMC
print("运行 MCMC...")
sampler.run_mcmc(pos, 5000, progress=True)

# 获取样本
samples = sampler.get_chain(discard=1000, thin=15, flat=True)

# 计算结果
V_mcmc, k_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(samples, [16, 50, 84], axis=0)))

print(f"V: {V_mcmc[0]:.3f} +{V_mcmc[1]:.3f} -{V_mcmc[2]:.3f}")
print(f"k: {k_mcmc[0]:.3f} +{k_mcmc[1]:.3f} -{k_mcmc[2]:.3f}")

# 计算半衰期
t_half_samples = np.log(2) / samples[:, 1]
t_half_mcmc = np.percentile(t_half_samples, [16, 50, 84])
print(f"半衰期: {t_half_mcmc[1]:.2f} +{t_half_mcmc[2]-t_half_mcmc[1]:.2f} -{t_half_mcmc[1]-t_half_mcmc[0]:.2f}")

# 绘制角图
fig = corner.corner(samples, labels=["V", "k"], truths=[V_mcmc[0], k_mcmc[0]])
plt.show()

# 绘制拟合曲线及其置信区间
t_fit = np.linspace(0, 20, 100)
plt.figure(figsize=(10, 6))
for i in range(100):
    sample = samples[np.random.randint(len(samples))]
    plt.plot(t_fit, model(t_fit, sample[0], sample[1]), "r-", alpha=0.1)
plt.errorbar(time, concentration, yerr=concentration*0.1, fmt=".k", capsize=0)
plt.xlabel("时间 (小时)")
plt.ylabel("浓度 (mg/L)")
plt.title("贝叶斯拟合结果")
plt.show()

# 使用所有点计算统计量
V_median, k_median = np.median(samples, axis=0)
residuals_all = concentration - model(time, V_median, k_median)

# R-squared
ss_tot = np.sum((concentration - np.mean(concentration))**2)
ss_res = np.sum(residuals_all**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.4f}")

# Shapiro-Wilk 正态性检验
_, p_value = stats.shapiro(residuals_all)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")

# 残差图
plt.figure(figsize=(10, 6))
plt.scatter(time, residuals_all)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("时间 (小时)")
plt.ylabel("残差")
plt.title("残差图 (所有点)")
plt.show()

# Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(residuals_all, dist="norm", plot=plt)
plt.title("Q-Q 图 (所有点)")
plt.show()

'''
角图 (Corner Plot):

这是一个多维参数分布的可视化图。
对角线上显示每个参数(V和k)的边缘分布。
非对角线元素显示参数之间的联合分布。
这个图可以帮助我们理解参数之间的相关性和不确定性。
贝叶斯拟合结果图:

显示原始数据点和拟合曲线。
红色区域表示模型预测的95%置信区间。
黑点是实际观测数据,误差棒表示测量误差。
这个图直观地展示了模型对数据的拟合程度。
残差图:

x轴是时间,y轴是残差(观测值减去预测值)。
理想情况下,残差应该随机分布在y=0线周围。
这个图可以帮助我们检测模型是否存在系统性偏差。
Q-Q图 (Quantile-Quantile Plot):

用于检验残差是否服从正态分布。
x轴是理论分位数,y轴是样本分位数。
如果点大致落在对角线上,说明残差近似正态分布。
偏离对角线则表明残差分布可能存在偏斜或异常值。
'''
