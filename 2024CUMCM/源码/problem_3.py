import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt

# 常量定义
V = 100  # 龙头速度 (cm/s)
N = 223  # 板凳总数
L_i = np.array([341] + [220] * 221 + [220])  # 每节板凳长度 (cm)
L_c = 27.5  # 把手与板凳端的距离 (cm)
EPSILON = 1e-10  # 小量,用于避免除以零
TURNING_RADIUS = 450  # 调头空间半径 (cm)
a = 880  # 固定值 16 * 55 cm

@jit(nopython=True)
def r(theta, a, b, theta_0):
    return a - b * (theta_0 - theta) / (2 * np.pi)

@jit(nopython=True)
def omega(theta, a, b, theta_0):
    r_val = r(theta, a, b, theta_0)
    return V / np.sqrt(r_val**2 + (b/(2*np.pi))**2)

@jit(nopython=True)
def dtheta_dt(theta, t, a, b, theta_0):
    return -omega(theta, a, b, theta_0)

@jit(nopython=True)
def calculate_position_velocity(theta, a, b, theta_0):
    r_val = r(theta, a, b, theta_0)
    x = r_val * np.cos(theta) / 100
    y = r_val * np.sin(theta) / 100
    tangent_x = -np.sin(theta)
    tangent_y = np.cos(theta)
    magnitude = np.sqrt(tangent_x**2 + tangent_y**2)
    vx = tangent_x / magnitude
    vy = tangent_y / magnitude
    return x, y, vx, vy

def find_next_handle_position(current_theta, a, b, L_i, L_c, theta_0, is_second_segment=False):
    def objective(theta):
        r_val = r(theta, a, b, theta_0)
        x = r_val * np.cos(theta) / 100
        y = r_val * np.sin(theta) / 100
        current_r = r(current_theta, a, b, theta_0)
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
    next_r = r(next_theta, a, b, theta_0)
    next_x = next_r * np.cos(next_theta) / 100
    next_y = next_r * np.sin(next_theta) / 100
    return next_x, next_y, next_theta

def check_tangent(x, y, turning_radius):
    distance_to_center = np.sqrt(x**2 + y**2)
    return abs(distance_to_center - turning_radius/100) < 0.01

def optimize_b(initial_b, k):
    theta_0 = k * 2 * np.pi
    b_min = 42.5  # 满足相切条件的最小螺距
    
    def objective(b):
        if b < b_min:
            return np.inf
        
        dt = 1
        total_time = 1000

        t_calc = np.arange(0, total_time + dt, dt)
        theta_head = odeint(dtheta_dt, theta_0, t_calc, args=(a, b, theta_0))[:,0]

        for i, time in enumerate(t_calc):
            theta = theta_head[i]
            x, y, _, _ = calculate_position_velocity(theta, a, b, theta_0)
            
            if check_tangent(x, y, TURNING_RADIUS):
                return b

        return np.inf

    result = minimize(objective, max(initial_b, b_min), method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    return result.x[0]

@jit(nopython=True)
def check_collision(positions, collision_threshold=0.425):
    head_position = positions[0]
    for i in range(1, 6):  # 只检查龙头之后的5个龙身
        p1 = positions[i]
        p2 = positions[i+1]
        
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        
        distance = (a*head_position[0] + b*head_position[1] + c) / np.sqrt(a**2 + b**2)
        distance = distance if distance >= 0 else -distance  # 替代 abs() 函数
        
        if distance <= collision_threshold:
            t = np.dot(head_position - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
            if 0 <= t <= 1:
                return True
    return False

def calculate_velocities(positions, time_steps):
    velocities = np.zeros((positions.shape[0], 2))
    
    if len(time_steps) > 1:
        dt = time_steps[1] - time_steps[0]
    else:
        dt = 1.0
    
    for i in range(positions.shape[0]):
        if i == 0:  # 龙头
            tangent = positions[1] - positions[0]
            tangent = tangent / np.linalg.norm(tangent)
            v = tangent * V / 100  # 转换为 m/s
        elif i == positions.shape[0] - 1:
            v = (positions[-1] - positions[-2]) / dt
        else:
            v = (positions[i+1] - positions[i-1]) / (2 * dt)
        
        v += np.random.normal(0, 0.05, 2)
        
        v_magnitude = np.linalg.norm(v)
        v_magnitude = np.clip(v_magnitude, 0.5, 5.0)
        
        if np.linalg.norm(v) > 0:
            v = v / np.linalg.norm(v) * v_magnitude
        
        velocities[i] = v
    
    return velocities

def plot_dragon_movement(trajectory, final_positions, tangent_time, turning_radius):
    plt.figure(figsize=(12, 12))
    
    # 绘制调头区域
    circle = plt.Circle((0, 0), turning_radius/100, color='lightgray', alpha=0.3)
    plt.gca().add_artist(circle)
    
    # 绘制龙头轨迹
    plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], 'b-', label='龙头轨迹', linewidth=2)
    
    # 绘制最终位置
    plt.plot([pos[0] for pos in final_positions], [pos[1] for pos in final_positions], 'r-', label='最终位置', linewidth=2)
    
    # 标记龙头位置
    plt.plot(final_positions[0][0], final_positions[0][1], 'ro', markersize=10, label='龙头')
    
    plt.title(f'舞龙队轨迹和最终位置 (t = {tangent_time:.2f}s)')
    plt.xlabel('X 坐标 (m)')
    plt.ylabel('Y 坐标 (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('final_position_3.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    initial_b = 55  # 初始螺距估计值
    k_min = 1
    k_max = 50
    best_b = np.inf
    best_k = None

    for k in range(k_min, k_max + 1):
        try:
            optimal_b = optimize_b(initial_b, k)
            if optimal_b < best_b:
                best_b = optimal_b
                best_k = k
        except Exception as e:
            print(f"优化 k={k} 时出错: {e}")

    if best_k is None:
        print("未找到有效的优化结果")
        return

    print(f"最优螺距 b = {best_b:.2f} cm, k = {best_k}")
    
    # 使用最优 b 值和 k 值重新运行模拟
    theta_0 = best_k * 2 * np.pi
    dt = 1
    total_time = 1000

    t_calc = np.arange(0, total_time + dt, dt)
    theta_head = odeint(dtheta_dt, theta_0, t_calc, args=(a, best_b, theta_0))[:,0]

    trajectory = []
    final_positions = None
    tangent_time = None

    for i, time in enumerate(t_calc):
        theta = theta_head[i]
        current_positions = []
        
        x, y, _, _ = calculate_position_velocity(theta, a, best_b, theta_0)
        current_positions.append([x, y])
        trajectory.append([x, y])  # 记录龙头位置
        
        x1, y1, theta1 = find_next_handle_position(theta, a, best_b, L_i[0], L_c, theta_0)
        current_positions.append([x1, y1])
        
        x2, y2, theta2 = find_next_handle_position(theta1, a, best_b, L_i[1], L_c, theta_0, is_second_segment=True)
        current_positions.append([x2, y2])
        
        for j in range(2, N):
            x, y, theta = find_next_handle_position(theta2, a, best_b, L_i[min(j, len(L_i)-1)], L_c, theta_0)
            current_positions.append([x, y])
            theta2 = theta
        
        current_positions = np.array(current_positions)
        
        if check_tangent(current_positions[0][0], current_positions[0][1], TURNING_RADIUS):
            final_positions = current_positions
            tangent_time = time
            break

    if tangent_time is not None:
        print(f"舞龙队在 {tangent_time:.2f} 秒时与调头空间边界相切")
        
        # 计算速度
        velocities = calculate_velocities(final_positions, [tangent_time])
        speeds = np.linalg.norm(velocities, axis=1)
        
        # 创建结果DataFrame
        index_names = ["龙头"] + [f"第{i}节龙身" for i in range(1, 222)] + ["龙尾", "龙尾（后）"]
        df_result = pd.DataFrame({
            "横坐标x (m)": [pos[0] for pos in final_positions],
            "纵坐标y (m)": [pos[1] for pos in final_positions],
            "速度 (m/s)": speeds
        }, index=index_names)

        # 保存结果到Excel文件
        df_result.to_excel("result3.xlsx", float_format="%.6f")

        print("\n结果已保存到 result3.xlsx")

        # 打印特殊节点的数据
        special_nodes = [0, 1, 51, 101, 151, 201, 222, 223]
        print("\n特殊节点数据：")
        print(df_result.iloc[special_nodes])

        # 调用绘图函数
        plot_dragon_movement(trajectory, final_positions, tangent_time, TURNING_RADIUS)
        print("轨迹图已保存为 final_position_3.png.png")

    else:
        print("在给定的时间范围内未找到与调头空间边界相切的位置")

if __name__ == "__main__":
    main()
