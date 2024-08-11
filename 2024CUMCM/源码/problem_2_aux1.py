import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
L_c = 27.5  # 把手与板凳端的距离 (cm)

def calculate_bench_corners(start, end, width=0.3):
    """
    计算板凳的四个角点坐标,考虑L_c
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 计算垂直于板凳长度方向的单位向量
    perpendicular_x = -np.sin(angle)
    perpendicular_y = np.cos(angle)
    
    # 使用length计算L_c在x和y方向的分量
    lc_x = L_c/100 * dx / length
    lc_y = L_c/100 * dy / length
    
    # 计算板凳实际起点和终点
    start_actual = (start[0] - lc_x, start[1] - lc_y)
    end_actual = (end[0] + lc_x, end[1] + lc_y)
    
    # 计算四个角点
    corner1 = (start_actual[0] - perpendicular_x * width/2, start_actual[1] - perpendicular_y * width/2)
    corner2 = (start_actual[0] + perpendicular_x * width/2, start_actual[1] + perpendicular_y * width/2)
    corner3 = (end_actual[0] + perpendicular_x * width/2, end_actual[1] + perpendicular_y * width/2)
    corner4 = (end_actual[0] - perpendicular_x * width/2, end_actual[1] - perpendicular_y * width/2)
    
    return [corner1, corner2, corner3, corner4]

def draw_bench(ax, start, end, width=0.3):
    """
    绘制一个板凳
    """
    corners = calculate_bench_corners(start, end, width)
    bench = plt.Polygon(corners, closed=True, facecolor='lightblue', edgecolor='blue')
    ax.add_patch(bench)

def main():
    # 从 result2.xlsx 读取数据
    df = pd.read_excel('result2.xlsx', index_col=0)
    
    positions = df[['横坐标x (m)', '纵坐标y (m)']].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 绘制每个板凳
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i+1]
        draw_bench(ax, start, end)
    
    # 标记特殊点
    special_nodes = [0, 1, 51, 101, 151, 201, 223]
    node_names = ["龙头", "第1节龙身", "第51节龙身", "第101节龙身", "第151节龙身", "第201节龙身", "龙尾（后）"]
    
    for node, name in zip(special_nodes, node_names):
        x, y = positions[node]
        ax.plot(x, y, 'ro', markersize=8)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
    
    # 设置图形属性
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X 坐标 (m)')
    ax.set_ylabel('Y 坐标 (m)')
    ax.set_title('板凳龙最终位置示意图 (考虑L_c)')
    ax.grid(True)
    
    # 保存图形
    plt.savefig('final_position_2_aux1.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
