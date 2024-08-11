import sympy as sp
import matplotlib.pyplot as plt

# Apply the desired plot style
plt.style.use('seaborn-v0_8-whitegrid')

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

# Function to compute total population for each year
def compute_total_population(distributions):
    return [sum(dist) for dist in distributions]

# Set the number of time steps to compute
k = 5

# r values from 0 to 1 with step size 0.1
r_values = [i / 10.0 for i in range(9)]  # r = 0.0 to 0.8

# Create a figure with a 4x2 layout
fig, axs = plt.subplots(4, 2, figsize=(12, 16))  # Adjusted size for better visibility
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

# Initialize a counter for valid plots
valid_plot_count = 0

# Plotting bar charts for each r value
for r_val in r_values:
    # Skip r = 0
    if r_val == 0:
        continue

    # Substitute the value of r into the Leslie matrix L
    L_r = L.subs(r, r_val)
    
    # Calculate the population distribution vector for each year
    population_distributions = compute_population_distribution(x0, L_r, k)
    
    # Calculate the total population for each year
    total_population = compute_total_population(population_distributions)
    
    # Extract total population values for plotting
    total_population_values = [float(tp) for tp in total_population]

    # Plotting the bar chart for the current r value
    axs[valid_plot_count].bar(range(len(total_population_values)), total_population_values, color='skyblue', width=0.6)
    axs[valid_plot_count].set_xlabel('Year')
    axs[valid_plot_count].set_ylabel('Total Population')
    axs[valid_plot_count].set_title(f'Total Cattle Population Over 5 Years (r = {r_val})')
    axs[valid_plot_count].set_xticks(range(len(total_population_values)))
    axs[valid_plot_count].set_xticklabels(['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'])

    # Adding a warning line at y = 120 with some transparency
    axs[valid_plot_count].axhline(y=120, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Increment the valid plot counter
    valid_plot_count += 1

# Hide any unused subplots
for j in range(valid_plot_count, len(axs)):
    axs[j].axis('off')  # Turn off the axis for unused plots

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('r递增的羊群数量变化-直方图')
plt.show()