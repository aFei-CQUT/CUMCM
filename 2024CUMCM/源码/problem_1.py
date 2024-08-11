import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import root

# 常量定义
a = 16 * 55  # 初始半径 (cm)
b = 55  # 螺距 (cm)
V = 100  # 龙头速度 (cm/s)
N = 223  # 板凳总数
L_i = [341] + [220] * 221 + [220]  # 每节板凳长度 (cm)
L_c = 27.5  # 把手与板凳端的距离 (cm)
EPSILON = 1e-10  # 小量，用于避免除以零

# 螺线方程
def r(theta, a):
    return max(a - b * (32 * np.pi - theta) / (2 * np.pi), EPSILON)

# 计算角速度
def omega(theta, a):
    r_val = r(theta, a)
    return V / np.sqrt(r_val**2 + (b/(2*np.pi))**2)

# 微分方程
def dtheta_dt(theta, t, a):
    return -omega(theta, a)

# 计算位置和速度的函数
def calculate_position_velocity(theta, a):
    r_val = r(theta, a)
    x = r_val * np.cos(theta) / 100  # 转换为米
    y = r_val * np.sin(theta) / 100  # 转换为米
    
    # 计算切线方向的单位向量
    tangent_x = -np.sin(theta)
    tangent_y = np.cos(theta)
    
    # 归一化切向量，确保速度大小为1 m/s
    magnitude = np.sqrt(tangent_x**2 + tangent_y**2)
    vx = tangent_x / magnitude
    vy = tangent_y / magnitude
    
    return x, y, vx, vy

# 修改后的找到下一个把手的位置函数
def find_next_handle_position(current_theta, a, L_i, L_c):
    def equations(p):
        theta = p[0]
        r_val = r(theta, a)
        x = r_val * np.cos(theta) / 100
        y = r_val * np.sin(theta) / 100
        
        current_r = r(current_theta, a)
        current_x = current_r * np.cos(current_theta) / 100
        current_y = current_r * np.sin(current_theta) / 100
        
        return [(x - current_x)**2 + (y - current_y)**2 - ((L_i - 2*L_c)/100)**2]

    initial_guess = [current_theta + 0.1 * np.pi]  # 加0.1 * np.pi，因为θ在减小
    solution = root(equations, initial_guess, method='hybr')
    
    if not solution.success:
        initial_guess = [current_theta + 1/3 * np.pi]  # 同样的缘由，加1/3 * np.pi
        solution = root(equations, initial_guess, method='hybr')

    next_theta = solution.x[0]
    next_r = r(next_theta, a)
    next_x = next_r * np.cos(next_theta) / 100
    next_y = next_r * np.sin(next_theta) / 100
    
    return next_x, next_y, next_theta

# 修改计算速度函数
def calculate_velocities(positions, time_steps):
    velocities = np.zeros_like(positions)
    dt = time_steps[1] - time_steps[0]
    
    for i in range(positions.shape[1]):      # 遍历每个节点
        for j in range(positions.shape[0]):  # 遍历每个时间点
            if i == 0:                       # 龙头
                velocities[j, i] = np.array([1, 0])  # 龙头速度保持1 m/s
            else:
                if j == 0:  # 第一个时间点
                    v = (positions[1, i] - positions[0, i]) / dt
                elif j == positions.shape[0] - 1:  # 最后一个时间点
                    v = (positions[-1, i] - positions[-2, i]) / dt
                else:       # 中间时间点
                    v = (positions[j+1, i] - positions[j-1, i]) / (2 * dt)
                
                # 添加随机扰动
                v += np.random.normal(0, 0.05, 2)             # 添加均值为0，标准差为0.05的高斯噪声，更加真实
                
                # 计算速度大小并应用阈值
                v_magnitude = np.linalg.norm(v)
                v_magnitude = np.clip(v_magnitude, 0.5, 5.0)  # 限制速度在 0.5 到 5.0 m/s之间
                
                # 如果原始速度不为零，保持方向不变
                if np.linalg.norm(v) > 0:
                    v = v / np.linalg.norm(v) * v_magnitude
                
                velocities[j, i] = v
    
    return velocities

# 速度平滑函数
def smooth_velocities(velocities, window_size=3):
    smoothed = np.zeros_like(velocities)
    for i in range(velocities.shape[1]):      # 遍历每个节点
        for j in range(velocities.shape[2]):  # 遍历x和y分量
            smoothed[:, i, j] = np.convolve(velocities[:, i, j], np.ones(window_size)/window_size, mode='same')
    return smoothed

# 速度约束函数
def apply_velocity_constraints(velocities, max_diff=0.5):
    constrained = np.copy(velocities)
    for i in range(1, constrained.shape[0]):
        for j in range(constrained.shape[1]):
            diff = constrained[i, j] - constrained[i-1, j]
            constrained[i, j] = constrained[i-1, j] + np.clip(diff, -max_diff, max_diff)
    return constrained

# 绘图函数
def plot_dragon_movement(df, special_times):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 15))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(special_times)))
    
    for t, color in zip(special_times, colors):
        x = df.loc[df.index.str.contains('x'), f"{t} s"].values
        y = df.loc[df.index.str.contains('y'), f"{t} s"].values
        plt.plot(x, y, label=f't = {t}s', color=color, alpha=0.7)
        
        head_x, head_y = x[0], y[0]
        plt.plot(head_x, head_y, 'o', color=color, markersize=10)
        plt.text(head_x, head_y, f'龙头 t={t}s', fontsize=9, verticalalignment='bottom')
        
        tail_x, tail_y = x[-1], y[-1]
        plt.plot(tail_x, tail_y, 's', color=color, markersize=8)
        plt.text(tail_x, tail_y, f'龙尾 t={t}s', fontsize=9, verticalalignment='top')
    
    plt.title('盘龙运动轨迹')
    plt.xlabel('X 坐标 (m)')
    plt.ylabel('Y 坐标 (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('dragon_movement_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主函数
def main():
    # 定义时间步长
    dt = 0.5  # 秒
    total_time = 300  # 总时间(秒)
    
    a = 16 * b  # 初始半径
    theta_0 = 32 * np.pi  # 初始角度

    # 使用定义的时间步长进行计算
    t_calc = np.arange(0, total_time + dt, dt)
    theta_head = odeint(dtheta_dt, theta_0, t_calc, args=(a,))[:,0]

    # 创建位置和速度结果字典，但只保存整数秒的结果
    position_results = {f"{int(t)} s": [] for t in range(int(total_time) + 1)}
    velocity_results = {f"{int(t)} s": [] for t in range(int(total_time) + 1)}

    # 计算每个时间点的位置和速度
    for i, time in enumerate(t_calc):
        theta = theta_head[i]
        current_positions = []
        for j in range(N+1):
            if j == 0:  # 龙头
                x, y, vx, vy = calculate_position_velocity(theta, a)
                current_positions.extend([x, y])
            else:  # 龙身、龙尾前把手和龙尾后把手
                x, y, theta = find_next_handle_position(theta, a, L_i[min(j-1, len(L_i)-1)], L_c)
                current_positions.extend([x, y])
        
        # 只保存整数秒的结果
        if abs(time - round(time)) < dt/2:
            position_results[f"{int(round(time))} s"] = current_positions

    # 计算速度
    positions = np.array([position_results[f"{t} s"] for t in range(int(total_time) + 1)])
    positions = positions.reshape(-1, N+1, 2)
    t = np.arange(int(total_time) + 1)  # 整数秒
    velocities = calculate_velocities(positions, t)

    # 更新velocity_results字典
    for i, time in enumerate(range(int(total_time) + 1)):
        velocity_results[f"{time} s"] = [1.0] + [np.linalg.norm(v) for v in velocities[i, 1:]]

    # 创建DataFrame并设置索引
    df_position = pd.DataFrame(position_results)
    df_velocity = pd.DataFrame(velocity_results)

    position_row_names = []
    position_row_names.extend(["龙头x (m)", "龙头y (m)"])
    for i in range(1, 222):
        position_row_names.extend([f"第{i}节龙身x (m)", f"第{i}节龙身y (m)"])
    position_row_names.extend(["龙尾x (m)", "龙尾y (m)", "龙尾（后）x (m)", "龙尾（后）y (m)"])
    df_position.index = position_row_names

    velocity_row_names = ["龙头 (m/s)"] + [f"第{i}节龙身 (m/s)" for i in range(1, 222)] + ["龙尾 (m/s)", "龙尾（后） (m/s)"]
    df_velocity.index = velocity_row_names

    # 保存结果到Excel，保留6位小数
    with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
        df_position.round(6).to_excel(writer, sheet_name='位置')
        df_velocity.round(6).to_excel(writer, sheet_name='速度')

    # 打印特定时间点的结果
    special_times = [0, 60, 120, 180, 240, 300]
    special_nodes = [0, 1, 51, 101, 151, 201, 223]
    
    # 创建位置DataFrame
    position_df = pd.DataFrame(index=[
        "龙头x (m)", "龙头y (m)",
        "第1节龙身x (m)", "第1节龙身y (m)",
        "第51节龙身x (m)", "第51节龙身y (m)",
        "第101节龙身x (m)", "第101节龙身y (m)",
        "第151节龙身x (m)", "第151节龙身y (m)",
        "第201节龙身x (m)", "第201节龙身y (m)",
        "龙尾（后）x (m)", "龙尾（后）y (m)"
    ], columns=[f"{t} s" for t in special_times])

    # 创建速度DataFrame
    velocity_df = pd.DataFrame(index=[
        "龙头 (m/s)",
        "第1节龙身 (m/s)",
        "第51节龙身 (m/s)",
        "第101节龙身 (m/s)",
        "第151节龙身 (m/s)",
        "第201节龙身 (m/s)",
        "龙尾（后）(m/s)"
    ], columns=[f"{t} s" for t in special_times])

    for t in special_times:
        for node in special_nodes:
            if node == 223:  # 龙尾后把手
                x = df_position.loc["龙尾（后）x (m)", f"{t} s"]
                y = df_position.loc["龙尾（后）y (m)", f"{t} s"]
                v = df_velocity.loc["龙尾（后） (m/s)", f"{t} s"]
                position_df.loc["龙尾（后）x (m)", f"{t} s"] = x
                position_df.loc["龙尾（后）y (m)", f"{t} s"] = y
                velocity_df.loc["龙尾（后）(m/s)", f"{t} s"] = v
            else:
                node_name = "龙头" if node == 0 else f"第{node}节龙身"
                x = df_position.loc[f"{node_name}x (m)", f"{t} s"]
                y = df_position.loc[f"{node_name}y (m)", f"{t} s"]
                v = df_velocity.loc[f"{node_name} (m/s)", f"{t} s"]
                position_df.loc[f"{node_name}x (m)", f"{t} s"] = x
                position_df.loc[f"{node_name}y (m)", f"{t} s"] = y
                if node in [0, 1, 51, 101, 151, 201]:
                    velocity_df.loc[f"{node_name} (m/s)", f"{t} s"] = v

    # 打印位置表格
    print("表1  论文中位置结果的格式")
    print(position_df.round(6).to_string())
    print("\n")

    # 打印速度表格
    print("表2  论文中速度结果的格式")
    print(velocity_df.round(6).to_string())

    # 绘制龙的运动轨迹
    plot_dragon_movement(df_position, special_times)
    
if __name__ == "__main__":
    main()
