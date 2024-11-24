import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.api import VAR

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 加载归一化的 CSV 文件
file_path = r'./res/data/data_normalized.csv'
data_normalized = pd.read_csv(file_path, header=None)

# 指定需要分析的指标的索引
indices = [6, 3, 4, 8]  # 广义货币（现价本币单位）、GNI（不变价本币单位）、按消费者价格指数衡量的通货膨胀（年通胀率）、实际利率 （%）
labels = ["货币供给量", "GNI", "通货膨胀率", "利率"]

# 获取指定索引的数据
data = data_normalized.iloc[indices].T  # 转置以便于后续处理
data.columns = labels  # 重命名列名

# 打印数据以检查
print("数据的形状:", data.shape)
print(data.head())  # 打印前几行数据

# 检查数据的平稳性
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print('---')

# 对每个变量进行ADF检验
for label in labels:
    print(f'检验 {label} 的平稳性:')
    adf_test(data[label])

# 协整检验
def cointegration_test(series1, series2):
    score, p_value, _ = coint(series1, series2)
    print('协整检验结果:')
    print(f'Cointegration Score: {score}, p-value: {p_value}')

# 检查所有变量之间的协整关系
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        print(f'正在检验协整关系: {labels[i]} 和 {labels[j]}')
        cointegration_test(data[labels[i]], data[labels[j]])

# 构建VAR模型
model = VAR(data)
results = model.fit(maxlags=5)
print(results.summary())

# Granger因果关系检验
def granger_causality_test(data):
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                test_result = results.test_causality(labels[i], labels[j], kind='f')
                print(f'Granger因果关系检验: {labels[i]} -> {labels[j]}')
                print(test_result.summary())
                print('---')

# 进行Granger因果关系检验
granger_causality_test(data)