import pandas as pd
import os
import chardet
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def process_csv_files(directory):
    # 创建一个列表来存储所有文件的结果
    all_files = []
    
    # 遍历目录中的所有CSV文件
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)        
            processed_results_1 = pd.read_csv(file_path, header=None, skiprows=4, encoding='utf-8',
                             quotechar='"', thousands=',', na_values=[''])
            
            # 获取索引为0和41的行
            row_0 = processed_results_1.iloc[0].tolist()
            row_41 = processed_results_1.iloc[41].tolist() if len(processed_results_1) > 41 else None
            
            # 将结果添加到列表中
            all_files.append(row_0)
            if row_41:
                all_files.append(row_41)
    
    # 将结果写入处理后的CSV文件
    pd.DataFrame(all_files).to_csv(r'.\res\data\processed_results_1.csv', index=False, header=False)
    
    # 检测文件编码
    with open(r'.\res\data\processed_results_1.csv', 'rb') as f:
        result = chardet.detect(f.read())
    
    # 使用Pandas读取处理后的CSV文件
    processed_results_1 = pd.read_csv(r'.\res\data\processed_results_1.csv', encoding=result['encoding'])
    
    # 提取数据行
    processed_results_2 = processed_results_1.iloc[0::2]
    
    # 将数据行写入新的CSV文件
    processed_results_2.to_csv(r'.\res\data\processed_results_2.csv', index=False, header=False)
    
    # 读取processed_results_2.csv文件
    data_df = pd.read_csv(r'.\res\data\processed_results_2.csv', header=None)
    
    # 提取数据对应的变量名
    name = data_df.iloc[:, 2]
    
    # 提取数据
    data = data_df.iloc[:, 38:-1].values
    # 将提取的数据保存到CSV文件
    pd.DataFrame(data).to_csv(r'.\res\data\data.csv', index=False, header=False)

    # 使用KNN填充缺失值
    imputer = KNNImputer(n_neighbors=5)  # 可以调整n_neighbors的值
    data_imputed = imputer.fit_transform(data)

    # 将使用KNN填充缺失值后的数据保存到CSV文件
    pd.DataFrame(data_imputed).to_csv(r'.\res\data\data_imputed.csv', index=False, header=False)

    # 创建MinMaxScaler实例
    scaler = MinMaxScaler()

    # 对每一行进行归一化
    data_normalized = scaler.fit_transform(data_imputed.T).T  # Transpose to normalize rows, then transpose back

    # 将归一化后的数据保存到CSV文件
    pd.DataFrame(data_normalized).to_csv(r'.\res\data\data_normalized.csv', index=False, header=False)

    return name, data, data_imputed, data_normalized

# 设置目录路径
directory = r'.\res\csv'

# 处理CSV文件并获取数据
name, data, data_imputed, data_normalized = process_csv_files(directory)