import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numba import jit, vectorize

# 常量定义
a = 16 * 55  # 初始半径 (cm)
b = 55  # 螺距 (cm)
V = 100  # 龙头速度 (cm/s)
N = 223  # 板凳总数
L_i = np.array([341] + [220] * 221 + [220])  # 每节板凳长度 (cm)
L_c = 27.5  # 把手与板凳端的距离 (cm)
EPSILON = 1e-10  # 小量，用于避免除以零

@vectorize(['float64(float64, float64)'])
def r(theta, a):
    result = a - b * (32 * np.pi - theta) / (2 * np.pi)
    return np.maximum(result, EPSILON)

@jit(nopython=True)
def omega(theta, a):
    r_val = r(theta, a)
    return V / np.sqrt(r_val**2 + (b/(2*np.pi))**2)

@jit(nopython=True)
def dtheta_dt(theta, t, a):
    return -omega(theta, a)

@jit(nopython=True)
def calculate_position_velocity(theta, a):
    r_val = r(theta, a)
    x = r_val * np.cos(theta) / 100
    y = r_val * np.sin(theta) / 100
    tangent_x = -np.sin(theta)
    tangent_y = np.cos(theta)
    magnitude = np.sqrt(tangent_x**2 + tangent_y**2)
    vx = tangent_x / magnitude
    vy = tangent_y / magnitude
    return x, y, vx, vy

def find_next_handle_position(current_theta, a, L_i, L_c, is_second_segment=False):
    def objective(theta):
        r_val = r(theta, a)
        x = r_val * np.cos(theta) / 100
        y = r_val * np.sin(theta) / 100
        current_r = r(current_theta, a)
        current_x = current_r * np.cos(current_theta) / 100
        current_y = current_r * np.sin(current_theta) / 100
        distance = np.sqrt((x - current_x)**2 + (y - current_y)**2)
        return (distance - (L_i - 2*L_c)/100)**2

    initial_guess = current_theta + (0.05 if is_second_segment else 0.1 * np.pi)
    bounds = [(current_theta, current_theta + (np.pi/4 if is_second_segment else np.pi))]
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds)

    if not result.success:
        raise ValueError("无法找到下一个把手位置")

    next_theta = result.x[0]
    next_r = r(next_theta, a)
    next_x = next_r * np.cos(next_theta) / 100
    next_y = next_r * np.sin(next_theta) / 100
    return next_x, next_y, next_theta

@jit(nopython=True)
def line_equation(p1, p2):
    """计算通过两点的直线方程 ax + by + c = 0 的系数"""
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0]*p1[1] - p1[0]*p2[1]
    return a, b, c

@jit(nopython=True)
def point_to_line_distance(point, line_coeffs):
    """计算点到直线的距离"""
    a, b, c = line_coeffs
    x, y = point
    return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

@jit(nopython=True)
def check_collision(positions, collision_threshold=0.425):
    head_position = positions[0]
    for i in range(1, 6):  # 只检查龙头之后的5个龙身
        p1 = positions[i]
        p2 = positions[i+1]
        
        # 计算直线方程
        line_coeffs = line_equation(p1, p2)
        
        # 计算龙头到这条直线的距离
        distance = point_to_line_distance(head_position, line_coeffs)
        
        if distance <= collision_threshold:
            # 检查龙头是否在线段上
            t = np.dot(head_position - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
            if 0 <= t <= 1:
                return True
    return False

def calculate_velocities(positions, time_steps):
    velocities = np.zeros_like(positions, dtype=np.float64)
    
    if len(time_steps) > 1:
        dt = time_steps[1] - time_steps[0]
    else:
        dt = 1.0
    
    for i in range(positions.shape[1]):
        for j in range(positions.shape[0]):
            if j == 0:  # 龙头
                if i < positions.shape[1] - 1:
                    tangent = positions[0, i+1] - positions[0, i]
                else:
                    tangent = positions[0, i] - positions[0, i-1]
                tangent = tangent / np.linalg.norm(tangent)
                velocities[j, i] = tangent
            else:
                if positions.shape[0] == 1:
                    v = np.array([1.0, 0.0])
                elif j == positions.shape[0] - 1:
                    v = (positions[-1, i] - positions[-2, i]) / dt
                else:
                    v = (positions[j+1, i] - positions[j-1, i]) / (2 * dt)
                
                v = v.astype(np.float64)
                v += np.random.normal(0, 0.05, 2)
                
                v_magnitude = np.linalg.norm(v)
                v_magnitude = np.clip(v_magnitude, 0.5, 5.0)
                
                if np.linalg.norm(v) > 0:
                    v = v / np.linalg.norm(v) * v_magnitude
                
                velocities[j, i] = v
    
    return velocities

def smooth_velocities(velocities, window_size=3):
    if velocities.ndim == 2:
        smoothed = np.zeros_like(velocities)
        for j in range(velocities.shape[1]):
            smoothed[:, j] = np.convolve(velocities[:, j], np.ones(window_size)/window_size, mode='same')
    elif velocities.ndim == 3:
        smoothed = np.zeros_like(velocities)
        for i in range(velocities.shape[1]):
            for j in range(velocities.shape[2]):
                smoothed[:, i, j] = np.convolve(velocities[:, i, j], np.ones(window_size)/window_size, mode='same')
    else:
        raise ValueError("Unexpected shape for velocities array")
    return smoothed

def apply_velocity_constraints(velocities, max_diff=0.5):
    constrained = np.copy(velocities)
    for i in range(1, constrained.shape[0]):
        for j in range(constrained.shape[1]):
            diff = constrained[i, j] - constrained[i-1, j]
            constrained[i, j] = constrained[i-1, j] + np.clip(diff, -max_diff, max_diff)
    return constrained

def plot_dragon_movement(final_positions, collision_time, special_nodes, node_names):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 15))
    
    x_coords = [pos[0] for pos in final_positions]
    y_coords = [pos[1] for pos in final_positions]
    
    plt.plot(x_coords, y_coords, 'b-', label='龙身', alpha=0.7)
    
    for node, name in zip(special_nodes, node_names):
        x, y = final_positions[node]
        if node == 0:
            plt.plot(x, y, 'ro', markersize=10, label='龙头')
            plt.text(x, y, f'{name} t={collision_time:.6f}s', fontsize=9, verticalalignment='bottom')
        elif node == special_nodes[-1]:
            plt.plot(x, y, 'go', markersize=10, label='龙尾')
            plt.text(x, y, f'{name} t={collision_time:.6f}s', fontsize=9, verticalalignment='top')
        else:
            plt.plot(x, y, 'bo', markersize=6)
            plt.text(x, y, name, fontsize=8, verticalalignment='top')
    
    plt.title(f'舞龙队最终位置 (t = {collision_time:.6f}s)')
    plt.xlabel('X 坐标 (m)')
    plt.ylabel('Y 坐标 (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('final_position_2.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dt = 0.5  # 时间步长
    total_time = 500  # 总时间(秒)
    
    a = 16 * b
    theta_0 = 32 * np.pi

    t_calc = np.arange(0, total_time + dt, dt)
    theta_head = odeint(dtheta_dt, theta_0, t_calc, args=(a,))[:,0]

    collision_time = None
    final_positions = None

    for i, time in enumerate(t_calc):
        theta = theta_head[i]
        current_positions = []
        
        x, y, _, _ = calculate_position_velocity(theta, a)
        current_positions.append([x, y])
        
        x1, y1, theta1 = find_next_handle_position(theta, a, L_i[0], L_c)
        current_positions.append([x1, y1])
        
        x2, y2, theta2 = find_next_handle_position(theta1, a, L_i[1], L_c, is_second_segment=True)
        current_positions.append([x2, y2])
        
        for j in range(2, N):
            x, y, theta = find_next_handle_position(theta2, a, L_i[min(j, len(L_i)-1)], L_c)
            current_positions.append([x, y])
            theta2 = theta
        
        current_positions = np.array(current_positions)
        if check_collision(current_positions, collision_threshold=0.3):
            collision_time = time
            final_positions = current_positions
            break

        final_positions = current_positions

    if collision_time is not None:
        print(f"舞龙队在 {collision_time:.6f} 秒时无法继续盘入")
        
        # 计算最终速度
        final_positions_array = np.array(final_positions, dtype=np.float64)
        final_velocities = calculate_velocities(final_positions_array.reshape(1, -1, 2), np.array([collision_time]))
        
        # 应用平滑和约束
        final_velocities = smooth_velocities(final_velocities.squeeze())
        final_velocities = apply_velocity_constraints(final_velocities)
        
        # 确保 final_velocities 是 2D 数组
        if final_velocities.ndim == 1:
            final_velocities = final_velocities.reshape(-1, 2)

        # 创建结果DataFrame
        df_result = pd.DataFrame({
            "横坐标x (m)": [pos[0] for pos in final_positions],
            "纵坐标y (m)": [pos[1] for pos in final_positions],
            "速度 (m/s)": [1.0 if i == 0 else np.linalg.norm(vel) for i, vel in enumerate(final_velocities)]
        })

        # 设置索引
        index_names = ["龙头"] + [f"第{i}节龙身" for i in range(1, 222)] + ["龙尾", "龙尾（后）"]
        df_result.index = index_names

        # 保存结果到Excel
        df_result.round(6).to_excel('result2.xlsx')

        # 创建位置和速度表格
        special_nodes = [0, 1, 51, 101, 151, 201, 223]
        node_names = ["龙头", "第1节龙身", "第51节龙身", "第101节龙身", "第151节龙身", "第201节龙身", "龙尾（后）"]
        
        position_data = []
        velocity_data = []
        
        for node, name in zip(special_nodes, node_names):
            x, y = final_positions[node]
            v = 1.0 if node == 0 else df_result.loc[index_names[node], "速度 (m/s)"]
            position_data.extend([x, y])
            velocity_data.append(v)
            print(f"{name}: 位置: ({x:.6f}, {y:.6f}), 速度: {v:.6f}")

        # 创建位置表格
        df_position = pd.DataFrame({
            f"{collision_time:.6f} s": position_data
        }, index=[f"{name}{coord}" for name in node_names for coord in ['x (m)', 'y (m)']])
        
        # 创建速度表格
        df_velocity = pd.DataFrame({
            f"{collision_time:.6f} s": velocity_data
        }, index=[f"{name} (m/s)" for name in node_names])

        # 打印表格
        print("\n位置表格：")
        print(df_position.round(6))
        print("\n速度表格：")
        print(df_velocity.round(6))

        # 调用绘图函数
        plot_dragon_movement(final_positions, collision_time, special_nodes, node_names)
    else:
        print("在给定的时间范围内未发生碰撞")

if __name__ == "__main__":
    main()
