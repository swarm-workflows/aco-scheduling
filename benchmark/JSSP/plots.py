# %%
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
# %%
# Read the CSV file
df = pd.read_csv('results.csv')

# Filter df with column optimal=1
df_optimal = df[df['optimal'] == 1]
# Create a 3D scatter plot
fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111, projection='3d')

# Assign columns to axes
x = df_optimal['jobs']
y = df_optimal['machines']
z = df_optimal['time']

# Plotting
ax.scatter(x, y, z, s=z * 10, c=z, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Labeling axes
ax.set_xlabel('Jobs')
ax.set_ylabel('Machines')
ax.set_zlabel('Time')

# Show plot
plt.show()

# %%

df_filtered = df[df['optimal'] == 0]

# Plotting
fig, ax = plt.subplots()

# Scatter plots
# ax.scatter(df_filtered.index, df_filtered['opt_val'], color='blue', label='opt_val')
# ax.scatter(df_filtered.index, df_filtered['ortools_opt'], color='red', label='ortools_opt')

ax.scatter(df_optimal.index,
           df_optimal['opt_val'] / df_optimal['ortools_opt'] * 100,
           color="red",
           label="Optimal (OR-Tools)")
ax.scatter(df_filtered.index,
           df_filtered['opt_val'] / df_filtered['ortools_opt'] * 100,
           color="blue",
           label="Not optimal (OR-Tools)")

# Draw vertical dashed lines
# for i in df_filtered.index:
#     ax.plot([i, i], [df_filtered.loc[i, 'opt_val'], df_filtered.loc[i, 'ortools_opt']], 'k--')

# Labeling
ax.set_xlabel('Instance')
ax.set_ylabel('Percentage (%)')
ax.legend()

plt.show()
# %%
# Read the CSV file
df = pd.read_csv('results.csv')

# Create a 3D scatter plot
fig = plt.figure(figsize=(14, 6), tight_layout=True)

# First subplot for 3D scatter
ax = fig.add_subplot(121, projection='3d')  # Changed to 121 to accommodate two subplots

# Assign columns to axes
x = df['jobs']
y = df['machines']
z = df['time']

# Plotting 3D scatter
ax.scatter(x, y, z, s=z * 20, c=z, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Labeling axes for 3D scatter
ax.set_xlabel('Jobs')
ax.set_ylabel('Machines')
ax.set_zlabel('Time')

# Second subplot for histogram
ax2 = fig.add_subplot(122)  # 122 places this subplot to the right of the first one

# Plotting histogram
ax2.hist(z, bins=20, color='skyblue', edgecolor='black')

# Labeling axes for histogram
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Time')

# Show plot
plt.show()
# %%
