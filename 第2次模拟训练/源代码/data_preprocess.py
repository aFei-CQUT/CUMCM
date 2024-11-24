import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from sklearn.neighbors import LocalOutlierFactor

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从Excel表读取数据的函数
def read_data(file_path, sheet_names):
    data_dict = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path, header=None, sheet_name=sheet)
        time = []
        concentration = []
        for i in range(0, df.shape[0], 2):  # 提取时间和浓度
            time.extend(df.iloc[i, 1:].tolist())
            concentration.extend(df.iloc[i + 1, 1:].tolist())
        data_dict[sheet] = pd.DataFrame({'time': time, 'concentration': concentration})
    return data_dict

# 使用各种方法检测异常值的函数
def detect_outliers(data):
    def detect_outliers_zscore(data, window=3, threshold=2.5):
        rolling_mean = data.rolling(window=window, center=True).mean()
        rolling_std = data.rolling(window=window, center=True).std()
        z_scores = (data - rolling_mean) / rolling_std
        return np.abs(z_scores) > threshold

    def detect_outliers_mad(data, threshold=3):
        median = np.median(data)
        mad = median_abs_deviation(data)
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    def detect_outliers_lof(data, n_neighbors=5):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        outlier_labels = lof.fit_predict(data.values.reshape(-1, 1))
        return outlier_labels == -1

    def detect_outliers_moving_window(data, window=5, threshold=1.5):
        rolling_median = data.rolling(window=window, center=True).median()
        rolling_std = data.rolling(window=window, center=True).std()
        return np.abs(data - rolling_median) > threshold * rolling_std

    outliers_zscore = detect_outliers_zscore(data['concentration'])
    outliers_mad = detect_outliers_mad(data['concentration'])
    outliers_lof = detect_outliers_lof(data['concentration'])
    outliers_moving_window = detect_outliers_moving_window(data['concentration'])
    
    return outliers_zscore, outliers_mad, outliers_lof, outliers_moving_window

# 绘制异常值的函数
def plot_outliers(data, outliers, title, ax):
    ax.plot(data['time'], data['concentration'], 'o-')
    ax.plot(data['time'][outliers], data['concentration'][outliers], 'ro')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('时间 (min)', fontsize=12)
    ax.set_ylabel('血药浓度 (mg/L)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# 可视化异常值检测结果的函数
def visualize_outlier_detection(data, outliers_combined):
    plt.figure(figsize=(12, 6))
    plt.plot(data['time'], data['concentration'], 'o-', label='原始数据')
    methods = ['Z-score', 'MAD', 'LOF', '移动窗口']
    colors = ['ro', 'go', 'bo', 'yo']
    for idx, (outliers, color, method) in enumerate(zip(outliers_combined, colors, methods)):
        plt.plot(data['time'][outliers], data['concentration'][outliers], color, label=f'{method}')
    plt.title('异常值检测方法汇总', fontsize=16)
    plt.xlabel('时间 (min)', fontsize=12)
    plt.ylabel('血药浓度 (mg/L)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.show()

# 清洗数据以去除异常值的函数
def clean_data(data, outliers_combined):
    outliers_combined = np.any(outliers_combined, axis=0)
    return data[~outliers_combined].reset_index(drop=True)

# 执行分析的主函数
def main():
    file_path = r'./data.xlsx'
    sheet_names = ['Sheet1', 'Sheet2']
    
    # 读取数据
    data_dict = read_data(file_path, sheet_names)
    
    for sheet, data in data_dict.items():
        print(f"处理 {sheet} 的数据:")
        
        # 检测异常值
        outliers_combined = detect_outliers(data)
        
        # 可视化异常值
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('异常值检测方法比较', fontsize=16)
        titles = ['Z-score方法', 'MAD方法', 'LOF方法', '移动窗口法']
        for ax, outliers, title in zip(axs.flatten(), outliers_combined, titles):
            plot_outliers(data, outliers, title, ax)
        plt.tight_layout()
        plt.show()
        
        # 可视化组合异常值结果
        visualize_outlier_detection(data, outliers_combined)
        
        # 通过去除异常值清洗数据
        data_cleaned = clean_data(data, outliers_combined)
        print(f"清洗后的数据:\n{data_cleaned}\n")

if __name__ == "__main__":
    main()