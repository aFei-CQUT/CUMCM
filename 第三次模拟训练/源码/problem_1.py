import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 加载 CSV 文件
file_path = r'./res/data/data_imputed.csv'
data = pd.read_csv(file_path, header=None)

# 指定需要绘制的指标的索引和对应的标签
indices = [6, 2, 4, 8]
labels = ["货币供给量", "收入", "利率", "通货膨胀率"]

# 获取指定索引的数据
data_to_plot = data.iloc[indices]

# 转置数据以便于绘图和统计计算
data_transposed = data_to_plot.T

# 确保年份的生成与数据的行数一致
years = range(2024 - len(data_transposed.index), 2024)

# 1. 绘制条形图
axes_bar = data_transposed.plot(kind='bar', subplots=True, layout=(2, 2), figsize=(12*1.2, 9*1.2), legend=False)

# 设置 x 轴刻度标签
for ax in axes_bar.flatten():
    ax.set_xticks(range(len(data_transposed.index)))    # 设置刻度位置
    ax.set_xticklabels(years)                           # 设置刻度标签为年份
    ax.tick_params(axis='x', rotation=45)               # 旋转刻度标签
    ax.xaxis.set_tick_params(labelbottom=True)          # 确保显示 x 轴标签

# 为每个子图设置标题
for ax, label in zip(axes_bar.flatten(), labels):
    ax.set_title(label)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)                        # 调整上下间距
plt.savefig('./res/png/1_bar_plot.png', dpi=300)
plt.show()

# 2. 箱线图
fig, axes = plt.subplots(2, 2, figsize=(12*1.2, 9*1.2))

for i, (ax, label) in enumerate(zip(axes.flatten(), labels)):
    ax.boxplot(data.iloc[indices[i]].values)
    ax.set_title(label)
    ax.set_xticks([1])
    ax.set_xticklabels([label])                         # 设置箱线图的 x 轴标签为指标名称

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)                        # 调整上下间距
plt.savefig('./res/png/2_box_plot.png', dpi=300)
plt.show()

# 3. 面积图
axes_area = data_transposed.plot(kind='area', subplots=True, layout=(2, 2), stacked=False, figsize=(12*1.2, 9*1.2), legend=False)

for ax in axes_area.flatten():
    ax.set_xticks(range(len(data_transposed.index)))    # 设置刻度位置
    ax.set_xticklabels(years)                           # 设置刻度标签为年份
    ax.tick_params(axis='x', rotation=45)               # 旋转刻度标签
    ax.axhline(0, color='black', linewidth=0.8)         # 添加水平线
    ax.xaxis.set_tick_params(labelbottom=True)          # 确保显示 x 轴标签

# 为每个子图设置标题
for ax, label in zip(axes_area.flatten(), labels):
    ax.set_title(label)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)                        # 调整上下间距
plt.savefig('./res/png/3_area_plot.png', dpi=300)
plt.show()

# 统计量计算
statistics = data_transposed.describe()                 # 获取描述性统计

# 计算偏度和峰度
skewness = data_transposed.skew()                      # 计算偏度
kurtosis = data_transposed.kurtosis()                  # 计算峰度

# 将偏度和峰度添加到统计量中
statistics.loc['skewness'] = skewness
statistics.loc['kurtosis'] = kurtosis

# 打印统计量
print(statistics)

# 相关性分析
correlation_pearson = data_transposed.corr(method='pearson')   # 皮尔逊相关系数
correlation_spearman = data_transposed.corr(method='spearman') # 斯皮尔曼等级相关系数

# 打印相关性矩阵
print("Pearson Correlation Coefficients:")
print(correlation_pearson)
print("\nSpearman Correlation Coefficients:")
print(correlation_spearman)
