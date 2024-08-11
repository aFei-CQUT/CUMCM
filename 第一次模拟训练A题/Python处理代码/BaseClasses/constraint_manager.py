from cattle import Cattle
import numpy as np

class ConstraintManager:
    def __init__(self, x0, birth_rates, survival_rates, years):
        self.x0 = x0
        self.birth_rates = birth_rates
        self.survival_rates = survival_rates
        self.years = years

    def create_cattle(self, params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        return Cattle(
            x0=self.x0,
            birth_rates=self.birth_rates,
            survival_rates=self.survival_rates,
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=self.years
        )

    def constraint1(self, params):
        # 牛舍数量大于牛数
        r, M, *_ = params
        cattle = self.create_cattle(params)
        _, xs = cattle.simulate()
        return np.array([(M / 200 + 130) - np.sum(x) for x in xs])
    
    def constraint2(self, params):
        # 牧草面积下限
        _, _, alpha, *_ = params
        cattle = self.create_cattle(params)
        _, xs = cattle.simulate()
        return np.array([alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:])) for x in xs])

    def constraint3(self, params):
        # 总面积上限
        _, _, alpha, beta1, beta2, beta3, beta4, gamma = params
        return np.array([200 - np.sum([alpha, beta1, beta2, beta3, beta4, gamma])])

    def constraint4(self, params):
        # 成年母牛数量下限（仅在最后一年检查）
        cattle = self.create_cattle(params)
        _, xs = cattle.simulate()
        return np.array([np.sum(xs[-1][2:]) - 50])

    def constraint5(self, params):
        # 成年母牛数量上限（仅在最后一年检查）
        cattle = self.create_cattle(params)
        _, xs = cattle.simulate()
        return np.array([175 - np.sum(xs[-1][2:])])

    def constraint6(self, params):
        # 各年龄组牛的数量为非负整数
        cattle = self.create_cattle(params)
        _, xs = cattle.simulate()
        return np.array([np.all(x >= 0) and np.all(x == np.floor(x)) for x in xs])

    def constraint7(self, params):
        # 各种植面积为非负数
        _, _, alpha, beta1, beta2, beta3, beta4, gamma = params
        return np.array([alpha >= 0, beta1 >= 0, beta2 >= 0, beta3 >= 0, beta4 >= 0, gamma >= 0])

    def constraint8(self, params):
        # 小母牛出售率在0到1之间
        r, *_ = params
        return np.array([0 <= r <= 1])

    def get_constraints(self, params):
        # 获取所有约束条件并合并为一个数组
        constraints = np.concatenate([
            self.constraint1(params),
            self.constraint2(params),
            self.constraint3(params),
            self.constraint4(params),
            self.constraint5(params),
            self.constraint6(params),
            self.constraint7(params),
            self.constraint8(params)
        ])
        return constraints
