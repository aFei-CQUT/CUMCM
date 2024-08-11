import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False

# Apply the desired plot style
# plt.style.use('seaborn-v0_8-whitegrid')

# Initial population distribution vector x0
x0 = sp.Matrix([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

# Birth rate vector a
a = sp.Matrix([0, 0, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]).T  # Transposed to row vector

# Survival rate vector b
b = sp.Matrix([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]).T  # Transposed to row vector

# Define the symbolic variable r, the proportion of newborn heifers sold
r = sp.symbols('r')

# Construct the Leslie matrix L
n = len(x0)
L = sp.zeros(n)

# Set the first row of the Leslie matrix L considering the proportion r
L[0, :] = (1 - r) * a

# Assign survival rates to each age group in the Leslie matrix L
for i in range(1, n):
    L[i, i-1] = b[i-1]

# Function to compute population distribution vector over k years
def compute_population_distribution(x0, L, k):
    results = [x0.copy()]  # Initialize results list with the initial population
    x = x0
    for _ in range(k):
        x = L * x
        results.append(x.copy())  # Record population distribution each year
    return results

# Set the number of time steps to compute
k = 5

# Calculate the population distribution vector for each year
population_distributions = compute_population_distribution(x0, L, k)

# Simplify the total population expressions
total_populations_by_year = [sp.simplify(sum(dist)) for dist in population_distributions]

# Generate r values for the plot
r_values = np.linspace(0, 1, 100)

# Plotting the symbolic solutions for each year
plt.figure(figsize=(12, 8))

# Generate plots for each year's total population as a function of r
for year in range(k + 1):
    # Convert symbolic expression to a numeric function
    total_population_func = sp.lambdify(r, total_populations_by_year[year])
    total_population_values = [total_population_func(r_val) for r_val in r_values]
    plt.plot(r_values, total_population_values, label=f'第 {year} 年')

# 使用 LaTeX 语法设置标签和标题
plt.xlabel(r'新生小母牛出售比例 $(r)$')
plt.ylabel(r'牛群总数')
plt.title(r'5 年内总牛群数量随 $r$ 的变化')
plt.legend(title=r'     年份')
plt.grid(True)

# 设置横轴刻度间隔为 0.1
plt.xticks(np.arange(0, 1.1, 0.1))

plt.savefig('5 年内总牛群数量随r的变化')
plt.show()