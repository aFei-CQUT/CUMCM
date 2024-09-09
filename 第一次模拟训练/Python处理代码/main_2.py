import numpy as np
from scipy.optimize import differential_evolution


class Cattle:
    def __init__(self, x0, birth_rates, survival_rates, alpha, betas, gamma, r, M, years):
        self.x0 = np.array(x0)                    # 初始种群分布
        self.xs = [self.x0.copy()]                # 种群分布历史记录
        self.birth_rates = np.array(birth_rates)  
        self.survival_rates = np.array(survival_rates)
        self.alpha = alpha                        # 种植牧草
        self.betas = np.array(betas)              # 种植粮食
        self.gamma = gamma                        # 种植甜菜
        self.r = r                                # 小母牛出售率
        self.M = M                                # 贷款投资
        self.m = self.calculate_m(M)              # 年度固定成本
        self.years = years                        # 模拟年数
        self.n = len(x0)                          # 种群年龄组数量
        self.L_pp = self.create_L_pp()            # 种群转移矩阵 (所有)
        self.L_p = np.vstack([self.L_pp[0, :] * 0.5, self.L_pp[1:, :]])    # 种群转移矩阵 (公牛 或 母牛)
        self.L_r = np.vstack([self.L_p[0, :] * (1 - r), self.L_p[1:, :]])  # 种群转移矩阵 (考虑小母牛出售)
        self.y_1 = np.zeros(self.n); self.y_1[0] = 1                       # 用于选择第一年龄组的向量
        self.y_12 = np.zeros(self.n); self.y_12[-1] = 1                    # 用于选择最后一年龄组的向量
        self.y_1_2 = np.zeros(self.n); self.y_1_2[0:2] = 1                 # 用于选择前两个年龄组的向量
        self.y_3_12 = np.zeros(self.n); self.y_3_12[2:] = 1                # 用于选择3-12年龄组的向量
        self.reset_metrics()                      # 重置所有指标

    def calculate_m(self, M):
        # 计算年度还款金额
        return (M * 0.15 * (1 + 0.15)**10) / ((1 + 0.15)**10 - 1)

    def create_L_pp(self):
        # 创建种群转移矩阵
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        # 重置所有指标为初始状态
        self.alpha_values = np.array([self.alpha])
        self.betas_values = np.array([self.betas])
        self.gamma_values = np.array([self.gamma])
        self.q_betas_values = np.array([0])
        self.q_gammas_values = np.array([0])
        self.l_betas_values = np.array([0])
        self.l_gammas_values = np.array([0])
        self.w_betas_values = np.array([0])
        self.w_gammas_values = np.array([0])
        self.c_xiaomunius_values = np.array([0])
        self.c_damunius_values = np.array([0])
        self.c_betas_values = np.array([0])
        self.c_gammas_values = np.array([0])
        self.t_xiaomunius_values = np.array([0])
        self.t_damunius_values = np.array([0])
        self.t_betas_values = np.array([0])
        self.t_gammas_values = np.array([0])
        self.t_totals_values = np.array([0])
        self.c_workers_values = np.array([0])
        self.num_xiaogongniu_sales_values = np.array([0])
        self.num_xiaomuniu_sales_values = np.array([0])
        self.num_damuniu_sales_values = np.array([0])
        self.num_laomuniu_sales_values = np.array([0])
        self.w_xiaogongniu_values = np.array([0])
        self.w_xiaomuniu_values = np.array([0])
        self.w_damuniu_values = np.array([0])
        self.w_laomuniu_values = np.array([0])
        self.w_years_values = np.array([0])
        self.c_years_values = np.array([0])
        self.E_years_values = np.array([0])

    def calculate_penalty(self, x):
        # 计算给定状态的总惩罚
        penalty1 = min(0, (self.M / 200 + 130) - np.sum(x)) * 100                       # 满足约束 >=0 则取惩罚0
        penalty2 = min(0, self.alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:]))) * 100   # 满足约束 >=0 则取惩罚0
        penalty3 = min(0, 200 - np.sum([self.alpha, *self.betas, self.gamma])) * 100    # 满足约束 >=0 则取惩罚0
        penalty4 = min(0, np.sum(x[2:]) - 50) * 100                                     # 满足约束 >=0 则取惩罚0
        penalty5 = min(0, np.sum(x[2:]) - 175) * 100                                    # 满足约束 >=0 则取惩罚0
        
        return penalty1 + penalty2 + penalty3 + penalty4 + penalty5

    def simulate(self):
        # 模拟牧场运营
        total_penalty = 0
        last_x = self.xs[-1].copy()  # 记录上一个状态
    
        for year in range(self.years):
            x = self.xs[-1]
    
            # 计算约束条件
            constraint1 = (self.M / 200 + 130) - np.sum(x)                      # >= 0满足约束
            constraint2 = self.alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:]))  # >= 0满足约束
            constraint3 = 200 - np.sum([self.alpha, *self.betas, self.gamma])   # >= 0满足约束
            constraint4 = np.sum(x[2:]) - 50                                    # >= 0满足约束
            constraint5 = np.sum(x[2:]) - 175                                   # >= 0满足约束
    
            # 计算惩罚项
            penalty1 = min(0, constraint1) * 100                                  # 满足约束 >=0 则取惩罚0
            penalty2 = min(0, constraint2) * 100                                  # 满足约束 >=0 则取惩罚0
            penalty3 = min(0, constraint3) * 100                                  # 满足约束 >=0 则取惩罚0
            penalty4 = min(0, constraint4) * 100 if year == self.years - 1 else 0 # 满足约束 >=0 则取惩罚0
            penalty5 = min(0, constraint5) * 100 if year == self.years - 1 else 0 # 满足约束 >=0 则取惩罚0
    
            total_penalty += penalty1 + penalty2 + penalty3 + penalty4 + penalty5
            
            print(f"Year {year}: Profit = {self.calculate_total_profit():>12.3f}, Total Penalty = {total_penalty:>12.3f}")
    
            self.update_metrics(x)
    
            # 检查惩罚是否增加，如果增加则回退
            if total_penalty > 0 and total_penalty > self.calculate_penalty(last_x):
                self.xs[-1] = last_x.copy()  # 回退到上一个状态
                print("Penalty increased, reverting to last state.")
                
                # 调整参数
                current_penalty = self.calculate_penalty(last_x)
                if current_penalty > 1000:
                    self.M *= 1.1  # 增加M
                    print("Increasing M to address penalty.")
                elif current_penalty > 50:
                    self.alpha += 10  # 增加alpha
                    print("Increasing alpha to address penalty.")
                elif current_penalty > 20:
                    self.alpha -= 5
                    self.betas[0] -= 1
                    self.betas[1] -= 1
                    self.betas[2] -= 1
                    self.gamma -= 1
                    print("Decreasing alpha, beta1, beta2, beta3, and gamma to address penalty.")
            
                # 不使用break，允许继续循环下一个年份的模拟
                continue
    
            if year < self.years:
                self.update_x()
    
            last_x = x.copy()  # 更新上一个状态
    
        profit = self.calculate_total_profit()
        return profit - total_penalty, np.array(self.xs)

    def validate(self):
        # 验证最优参数
        for year in range(self.years):
            x = self.xs[-1]
            self.update_metrics(x)
            if year < self.years:
                self.update_x()
        
        return self.calculate_total_profit(), np.array(self.xs)

    def update_metrics(self, x):
        # 更新所有指标
        alpha = self.alpha
        beta1, beta2, beta3, beta4 = self.betas
        gamma = self.gamma
        
        q_beta = beta1 * 1.1 + beta2 * 0.9 + beta3 * 0.8 + beta4 * 0.6
        q_gamma = 1.5 * gamma
        
        l_beta = q_beta - 0.6 * np.sum(x[2:12])                         
        l_gamma = q_gamma - 0.7 * np.sum(x[2:12])
        
        w_beta = l_beta * 75 if l_beta > 0 else l_beta * 90
        w_gamma = l_gamma * 58 if l_gamma > 0 else l_gamma * 70
    
        self.alpha_values = np.append(self.alpha_values, alpha)
        self.betas_values = np.vstack([self.betas_values, [beta1, beta2, beta3, beta4]])
        self.gamma_values = np.append(self.gamma_values, gamma)
        self.q_betas_values = np.append(self.q_betas_values, q_beta)
        self.q_gammas_values = np.append(self.q_gammas_values, q_gamma)
        self.l_betas_values = np.append(self.l_betas_values, l_beta)
        self.l_gammas_values = np.append(self.l_gammas_values, l_gamma)
        self.w_betas_values = np.append(self.w_betas_values, w_beta)
        self.w_gammas_values = np.append(self.w_gammas_values, w_gamma)
    
        num_xiaogongniu_sales = np.floor(self.L_p @ x @ self.y_1).astype(int)
        num_xiaomuniu_sales = np.floor(self.L_p @ x @ self.y_1 * self.r).astype(int)
        num_damuniu_sales = np.floor(self.L_r @ x @ self.y_3_12).astype(int)
        num_laomuniu_sales = np.floor(self.L_r @ x @ self.y_12).astype(int)
    
        self.num_xiaogongniu_sales_values = np.append(self.num_xiaogongniu_sales_values, num_xiaogongniu_sales)
        self.num_xiaomuniu_sales_values = np.append(self.num_xiaomuniu_sales_values, num_xiaomuniu_sales)
        self.num_damuniu_sales_values = np.append(self.num_damuniu_sales_values, num_damuniu_sales)
        self.num_laomuniu_sales_values = np.append(self.num_laomuniu_sales_values, num_laomuniu_sales)
    
        self.w_xiaogongniu_values = np.append(self.w_xiaogongniu_values, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu_values = np.append(self.w_xiaomuniu_values, 40 * num_laomuniu_sales)
        self.w_damuniu_values = np.append(self.w_damuniu_values, 370 * num_damuniu_sales)
        self.w_laomuniu_values = np.append(self.w_laomuniu_values, 120 * num_laomuniu_sales)
    
        self.t_xiaomunius_values = np.append(self.t_xiaomunius_values, 10 * x @ self.y_1_2)
        self.t_damunius_values = np.append(self.t_damunius_values, 42 * x @ self.y_3_12)
        self.t_betas_values = np.append(self.t_betas_values, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas_values = np.append(self.t_gammas_values, 14 * gamma)
        self.t_totals_values = np.append(self.t_totals_values, 10 * x @ self.y_1_2 + 42 * x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)
    
        self.c_xiaomunius_values = np.append(self.c_xiaomunius_values, 500 * x @ self.y_1_2)
        self.c_damunius_values = np.append(self.c_damunius_values, 100 * x @ self.y_3_12)
        self.c_betas_values = np.append(self.c_betas_values, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas_values = np.append(self.c_gammas_values, 10 * gamma)
        self.c_workers_values = np.append(self.c_workers_values, 4000 if self.t_totals_values[-1] <= 5500 else 4000 + 1.2 * (self.t_totals_values[-1] - 5500))
    
        self.w_years_values = np.append(self.w_years_values, np.sum([self.w_xiaogongniu_values[-1], self.w_xiaomuniu_values[-1], self.w_damuniu_values[-1],
                                                                     self.w_laomuniu_values[-1], self.w_betas_values[-1], self.w_gammas_values[-1]]))
        
        self.c_years_values = np.append(self.c_years_values, np.sum([self.c_betas_values[-1], self.c_gammas_values[-1], self.c_xiaomunius_values[-1],
                                                                     self.c_damunius_values[-1], self.c_workers_values[-1], self.m]))
        
        self.E_years_values = np.append(self.E_years_values, self.w_years_values[-1] - self.c_years_values[-1])

    def update_x(self):
        # 更新种群分布
        x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum((np.floor(x)).astype(int), 0))
        
    def calculate_total_profit(self):
        # 计算总利润
        return np.sum(self.E_years_values)

if __name__ == '__main__':
    
    def objective_function(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
    
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=5
        )
        
        profit, xs = cattle.simulate()
        
        penalty = 0
        xs = np.array(xs)
    
        for t in range(len(xs)):
            x = xs[t]
            
            # 约束1：牛舍数量大于牛数
            constraint1 = (M / 200 + 130) - np.sum(x)
            if constraint1 < 0:
                penalty += abs(constraint1) * 1000
    
            # 约束2：牧草面积下限
            constraint2 = alpha - (2/3 * np.sum(x[:2]) + np.sum(x[2:]))
            if constraint2 < 0:
                penalty += abs(constraint2) * 1000
    
            # 约束3：土地面积上限
            constraint3 = 200 - (alpha + beta1 + beta2 + beta3 + beta4 + gamma)
            if constraint3 < 0:
                penalty += abs(constraint3) * 1000
    
            # 约束4：成年母牛数量下限（仅在最后一年检查）
            if t == len(xs) - 1:
                constraint4 = np.sum(x[2:]) - 50
                if constraint4 < 0:
                    penalty += abs(constraint4) * 2000  # 增加约束4的惩罚权重
    
            # 约束5：成年母牛数量上限（仅在最后一年检查）
            if t == len(xs) - 1:
                constraint5 = 175 - np.sum(x[2:])
                if constraint5 < 0:
                    penalty += abs(constraint5) * 1000
    
            # 约束6：各年龄组牛的数量为非负整数
            if not (np.all(x >= 0) and np.all(x == np.floor(x))):
                penalty += 1000
    
        # 约束7：各种植面积为非负数
        if not (alpha >= 0 and beta1 >= 0 and beta2 >= 0 and beta3 >= 0 and beta4 >= 0 and gamma >= 0):
            penalty += 1000
    
        # 约束8：小母牛出售率在0到1之间
        if not (0 <= r <= 1):
            penalty += 1000
    
        return -profit + penalty
    
    def validate_results(cattle, optimal_params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = optimal_params
        final_profit, xs_final = cattle.validate()
        
        print(f"\n年度利润:\n{cattle.E_years_values}")
        print(f"\nAlpha 值:\n{cattle.alpha_values}")
        print(f"\nBeta 值:\n{cattle.betas_values}")
        print(f"\nGamma 值:\n{cattle.gamma_values}")
        print(f"\n使用最优参数的最终利润:\n{final_profit}")
    
        penalty_check = 0
        constraint_violations = {i: 0 for i in range(1, 9)}
    
        for t in range(len(xs_final)):
            x = xs_final[t]
    
            if not ((M / 200 + 130) - np.sum(x) >= 0):
                penalty_check += 10000
                constraint_violations[1] += 1
    
            if not (alpha - (2/3 * np.sum(x[:2]) + np.sum(x[2:])) >= 0):
                penalty_check += 10000
                constraint_violations[2] += 1
    
            if not (200 - (alpha + beta1 + beta2 + beta3 + beta4 + gamma) >= 0):
                penalty_check += 10000
                constraint_violations[3] += 1
    
            if t == len(xs_final) - 1:
                if not (np.sum(x[2:]) - 50 >= 0):
                    penalty_check += 10000
                    constraint_violations[4] += 1
    
            if t == len(xs_final) - 1:
                if not (175 - np.sum(x[2:]) >= 0):
                    penalty_check += 10000
                    constraint_violations[5] += 1
    
            if not (np.all(x >= 0) and np.all(x == np.floor(x))):
                penalty_check += 10000
                constraint_violations[6] += 1
    
        if not (alpha >= 0 and beta1 >= 0 and beta2 >= 0 and beta3 >= 0 and beta4 >= 0 and gamma >= 0):
            penalty_check += 10000
            constraint_violations[7] += 1
    
        if not (0 <= r <= 1):
            penalty_check += 10000
            constraint_violations[8] += 1
    
        if penalty_check > 0:
            print("最终结果不符合所有约束条件，存在惩罚。")
            print(f"总惩罚: {penalty_check}")
            print("约束条件违反情况:")
            for constraint, violations in constraint_violations.items():
                if violations > 0:
                    print(f"约束{constraint}被违反 {violations} 次")
            print(f"\n特别注意: 约束4 (成年母牛数量下限) 被违反 {constraint_violations[4]} 次")
        else:
            print("最终结果符合所有约束条件。")
    
        return final_profit, constraint_violations

    bounds = [
        (0, 1),                     # r: 小母牛出售率
        (0, 1000000),               # M: 贷款投资
        (2/3*20+100, 200),          # alpha: 种植牧草
        (0, 20),                    # beta1: 种植粮食
        (0, 30),                    # beta2: 种植粮食
        (0, 30),                    # beta3: 种植粮食
        (0, 10),                    # beta4: 种植粮食
        (0, 200- (2/3*20+100))      # gamma: 种植甜菜
    ]
    
    def callback(xk, convergence):
        print(f"当前最优参数: {xk}")
        print(f"当前收敛度: {convergence}")
        print("---")

    print("开始优化过程...")
    
    result = differential_evolution(objective_function, bounds, callback=callback, disp=True, popsize=20, maxiter=1000)
    
    print("\n优化完成!")
    print(f"最优参数: \n{result.x}")
    print(f"最大利润: \n{-result.fun}")

    optimal_params = result.x
    r, M, alpha, beta1, beta2, beta3, beta4, gamma = optimal_params

    print("\n使用最优参数进行最终验证...")

    cattle_opt = Cattle(
        x0=np.ones(12) * 10,
        birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
        survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
        alpha=alpha,
        betas=[beta1, beta2, beta3, beta4],
        gamma=gamma,
        r=r,
        M=M,
        years=5
    )

    final_profit, constraint_violations = validate_results(cattle_opt, optimal_params)

    if constraint_violations[4] > 0:
        print("\n约束4 (成年母牛数量下限) 不满足,详细信息:")
        print(f"最后一年成年母牛数量: {np.sum(cattle_opt.xs[-1][2:])}")
        print("要求的最小数量: 50")
        print("建议: 可能需要调整参数以增加成年母牛数量,比如降低小母牛出售率或增加牧草面积。")