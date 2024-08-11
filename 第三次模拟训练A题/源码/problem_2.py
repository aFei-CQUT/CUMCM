import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 加载归一化的 CSV 文件
file_path = r'./res/data/data_normalized.csv'
data_normalized = pd.read_csv(file_path, header=None)

# 指定需要绘制的指标的索引和对应的标签
indices = [6, 3, 4, 8]  # M2、GNI、IR、RIR
labels = ["M2", "GNI", "IR", "RIR"]

# 获取指定索引的数据
data = data_normalized.iloc[indices].T  # 转置以便于后续处理
data.columns = labels

# 计算M2的变化率和GNI的变化量，并处理无穷大和NaN值
data['M2_change_rate'] = data['M2'].pct_change().replace([np.inf, -np.inf], np.nan)
data['GNI_change'] = data['GNI'].diff()

# 使用 KNNImputer 填补缺失值
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 应用 Yeo-Johnson 变换
pt = PowerTransformer(method='yeo-johnson', standardize=True)
data_transformed = pd.DataFrame(pt.fit_transform(data_imputed), columns=data_imputed.columns)

# 单变量回归模型
def single_variable_regression(X, y, label):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"\n{label} 回归模型摘要:")
    print(model.summary())
    return model

# 多元回归模型
def multivariate_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("\n多元回归模型摘要:")
    print(model.summary())
    return model

# 逆变换函数
def inverse_transform(y_pred_transformed, column_name):
    y_pred_original = pt.inverse_transform(data_transformed)[:, data_transformed.columns.get_loc('IR')]
    y_pred_original = y_pred_original * y_pred_transformed.std() + y_pred_transformed.mean()
    print(f"\n反变换后的预测 {column_name}:")
    print(y_pred_original)
    return y_pred_original

# 单变量回归
model_supply = single_variable_regression(data_transformed['M2_change_rate'], data_transformed['IR'], "货币供给")
model_demand = single_variable_regression(data_transformed['GNI_change'], data_transformed['IR'], "货币需求")
model_rir = single_variable_regression(data_transformed['RIR'], data_transformed['IR'], "实际利率")

# 多元回归
X_multivariate = data_transformed[['M2_change_rate', 'GNI_change', 'RIR']]
y_multivariate = data_transformed['IR']
model_multivariate = multivariate_regression(X_multivariate, y_multivariate)

# 预测
y_pred_supply = model_supply.predict(sm.add_constant(data_transformed['M2_change_rate']))
y_pred_demand = model_demand.predict(sm.add_constant(data_transformed['GNI_change']))
y_pred_rir = model_rir.predict(sm.add_constant(data_transformed['RIR']))
y_pred_multivariate = model_multivariate.predict(sm.add_constant(X_multivariate))

# 反变换预测结果
y_pred_supply_original = inverse_transform(y_pred_supply, "IR (M2)")
y_pred_demand_original = inverse_transform(y_pred_demand, "IR (GNI)")
y_pred_rir_original = inverse_transform(y_pred_rir, "IR (RIR)")
y_pred_multivariate_original = inverse_transform(y_pred_multivariate, "IR (Multivariate)")

# 集成学习模型函数
def fit_voting_model(X, y, y_label):
    # 定义多个基学习器
    model1 = LinearRegression()
    model2 = DecisionTreeRegressor(random_state=42)
    model3 = RandomForestRegressor(n_estimators=100, random_state=42)
    model4 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model5 = SVR()

    # 创建投票回归模型
    voting_model = VotingRegressor(estimators=[
        ('lr', model1),
        ('dt', model2),
        ('rf', model3),
        ('gb', model4),
        ('svr', model5)
    ])

    # 训练模型
    voting_model.fit(X, y)
    y_pred = voting_model.predict(X)
    
    print(f"\n{y_label} 投票模型 R-squared: {voting_model.score(X, y):.4f}")
    return y_pred

# 使用投票模型进行预测
y_pred_voting = fit_voting_model(X_multivariate, y_multivariate, "多元回归")

# 反变换投票模型的预测结果
y_pred_voting_original = inverse_transform(y_pred_voting, "IR (Voting)")

# 绘制原始数据和预测结果
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('利率预测对比', fontsize=16)

axs[0, 0].plot(data['IR'], label='实际IR')
axs[0, 0].plot(y_pred_supply_original, label='货币供给预测IR')
axs[0, 0].set_title('货币供给模型')
axs[0, 0].set_xlabel('时间')
axs[0, 0].set_ylabel('利率')
axs[0, 0].legend()

axs[0, 1].plot(data['IR'], label='实际IR')
axs[0, 1].plot(y_pred_demand_original, label='货币需求预测IR')
axs[0, 1].set_title('货币需求模型')
axs[0, 1].set_xlabel('时间')
axs[0, 1].set_ylabel('利率')
axs[0, 1].legend()

axs[1, 0].plot(data['IR'], label='实际IR')
axs[1, 0].plot(y_pred_rir_original, label='实际利率预测IR')
axs[1, 0].set_title('实际利率模型')
axs[1, 0].set_xlabel('时间')
axs[1, 0].set_ylabel('利率')
axs[1, 0].legend()

axs[1, 1].plot(data['IR'], label='实际IR')
axs[1, 1].plot(y_pred_multivariate_original, label='多元回归预测IR')
axs[1, 1].set_title('多元回归模型')
axs[1, 1].set_xlabel('时间')
axs[1, 1].set_ylabel('利率')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# 计算并打印 R-squared 值
print("\nR-squared 值:")
print(f"货币供给模型: {model_supply.rsquared:.4f}")
print(f"货币需求模型: {model_demand.rsquared:.4f}")
print(f"实际利率模型: {model_rir.rsquared:.4f}")
print(f"多元回归模型: {model_multivariate.rsquared:.4f}")

# =============================================================================
# data_normalized 或 data_imputed 中每一行数据含义(没有标签)
# 0 '消费者价格指数（2010 年 = 100）'
# 1 'GDP 增长率（年百分比）'
# 2 'GDP（不变价本币单位）'
# 3 'GNI（不变价本币单位）'
# 4 '按消费者价格指数衡量的通货膨胀（年通胀率）'
# 5 '广义货币增长（年度百分比）'
# 6 '广义货币（现价本币单位）'
# 7 '存款利率 (百分比)'
# 8 '实际利率 （%）'
# 9 '总失业人数（占劳动力总数的比例）（模拟劳工组织估计）'
# 10 '贷款利率 (百分比)'
# =============================================================================