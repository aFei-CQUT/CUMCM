import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.impute import KNNImputer
from statsmodels.tsa.api import VAR

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 加载归一化的 CSV 文件
file_path = r'./res/data/data_normalized.csv'
data_normalized = pd.read_csv(file_path, header=None)

# 指定需要分析的指标的索引
indices = [6, 3, 4, 8]  # 广义货币、GNI、通货膨胀率、实际利率
labels = ["货币供给量", "GNI", "通货膨胀率", "利率"]

# 获取指定索引的数据并转置
data = data_normalized.iloc[indices].T  
data.columns = labels  # 重命名列名

# 打印数据以检查
print("数据的形状:", data.shape)
print(data.head())  # 打印前几行数据

# 检查数据中的缺失值和无穷值
def check_for_infs_and_nans(data):
    if data.isna().any().any():
        print("数据中存在缺失值.")
    if np.isinf(data.values).any():
        print("数据中存在无穷值.")

# 处理无穷值和NaN值
def handle_inf_and_nan(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill().bfill()  # 使用 ffill() 和 bfill() 替代 fillna(method='ffill') 和 fillna(method='bfill')
    return data

# 对数变换
def log_transform(series):
    return np.log(series + 1)  # 加1以避免对数为负无穷

# 应用对数变换
for label in labels:
    data[label] = log_transform(data[label])

# 检查数据的平稳性
def adf_test(series):
    result = adfuller(series.dropna())
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

# 对不平稳数据进行差分处理
def difference(series):
    return series.diff().dropna()

# 检查是否需要多阶差分
def check_and_difference(data):
    differenced_data = pd.DataFrame(index=data.index)
    for label in labels:
        print(f'检验 {label} 的平稳性（差分前）:')
        adf_test(data[label])
        
        diff_count = 0
        while adfuller(data[label].dropna())[1] > 0.05:  # 如果p-value > 0.05，则数据不平稳
            print(f'{label} 不平稳，进行差分处理')
            data[label] = difference(data[label])
            diff_count += 1
            if diff_count > 5:  # 限制最多进行五阶差分
                print(f'{label} 需要多阶差分处理，超出最大差分次数')
                break
        
        differenced_data[label] = data[label]
        print(f'{label} 平稳性差分处理后:')
        adf_test(differenced_data[label])
        
    return differenced_data

# 处理无穷值和NaN值
data = handle_inf_and_nan(data)

differenced_data = check_and_difference(data)

# 处理缺失值（前向填补）
differenced_data = differenced_data.ffill()

# 再次检查缺失值
check_for_infs_and_nans(differenced_data)

# 如果仍有缺失值，使用KNN插补
if differenced_data.isna().any().any():
    print("数据中仍有缺失值，使用KNN插补")
    imputer = KNNImputer(n_neighbors=5)
    differenced_data_np = differenced_data.to_numpy()  # 转换为numpy数组以便KNN插补
    differenced_data_np = imputer.fit_transform(differenced_data_np)
    differenced_data = pd.DataFrame(differenced_data_np, columns=differenced_data.columns, index=differenced_data.index)

# 再次检查数据
check_for_infs_and_nans(differenced_data)

# 确认处理后的数据
print("处理后的数据形状:", differenced_data.shape)
print(differenced_data.head())

# 绘制 ACF 和 PACF 图
def plot_acf_pacf(data, labels):
    fig, axes = plt.subplots(len(labels), 2, figsize=(14, 3 * len(labels)))
    max_lag = min(len(data) // 2 - 1, 20)  # 选择滞后期数
    for i, label in enumerate(labels):
        # ACF 图
        plot_acf(data[label].dropna(), ax=axes[i, 0], lags=max_lag, title=f'ACF of {label}')
        # PACF 图
        plot_pacf(data[label].dropna(), ax=axes[i, 1], lags=max_lag, title=f'PACF of {label}')
    plt.tight_layout()
    plt.show()

plot_acf_pacf(differenced_data, labels)

# 使用 VAR 模型进行多维时间序列建模
def fit_var_model(data):
    model = VAR(data)
    # 计算合适的maxlags
    n_obs = len(data)
    n_vars = len(data.columns)
    max_lags = min(int(np.floor((n_obs - 2) / (2 * n_vars))), 15)
    print(f"使用的最大滞后阶数: {max_lags}")
    model_fitted = model.fit(maxlags=max_lags, ic='aic')
    return model_fitted

# 训练VAR模型
var_model = fit_var_model(differenced_data)

# 预测未来走势
def forecast_var_model(model, data, steps=5):
    print("预测数据的形状:", data.shape)
    print("检查数据中的缺失值和无穷值:")
    check_for_infs_and_nans(data)

    forecast = model.forecast(data.values[-model.k_ar:], steps=steps)
    
    # 生成预测的日期索引
    last_date = data.index[-1]
    if isinstance(last_date, pd.Timestamp):
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    else:
        forecast_index = range(len(data), len(data) + steps)
    
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)
    return forecast_df

forecast_df = forecast_var_model(var_model, differenced_data, steps=5)
print('VAR模型未来 5 步预测:')
print(forecast_df)
