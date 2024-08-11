import os

# 获取当前工作目录
current_directory = os.getcwd()

# 列出当前目录中的所有文件和子目录
files_and_dirs = os.listdir(current_directory)

# 定义输出文件的路径
output_file_path = os.path.join(current_directory, '当前目录列表.txt')

# 将文件和目录列表写入文本文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("当前目录: {}\n".format(current_directory))
    file.write("目录列表:\n")
    for item in files_and_dirs:
        file.write("{}\n".format(item))

print(f"文件和目录列表已保存到 '{output_file_path}'。")
