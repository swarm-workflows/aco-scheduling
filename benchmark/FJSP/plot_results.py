# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Read CSV data
df = pd.read_csv('./result.csv')

# %%
# Plot 1: Makespans
plt.figure(figsize=(8, 4))
x = range(len(df['case']))
width = 0.25
plt.bar([i - width for i in x], df['LRU_makespan'], width, label='LRU')
plt.bar(x, df['OR_makespan'], width, label='OR-Tools')
plt.bar([i + width for i in x], df['ACO_makespan'], width, label='ACO')
plt.title('Makespan Comparison')
plt.xlabel('Test Case')
plt.ylabel('Makespan')
plt.xticks(x, df['case'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("./makespans.pdf")
# plt.savefig('benchmark_fjsp/makespans.png')

# %%
# Plot 2: Times
plt.figure(figsize=(8, 4))
x = range(len(df['case']))
width = 0.25
plt.bar([i - width for i in x], df['LRU_time'], width, label='LRU')
plt.bar(x, df['OR_time'], width, label='OR-Tools')
plt.bar([i + width for i in x], df['ACO_time'], width, label='ACO')
plt.title('Execution Time Comparison')
plt.xlabel('Test Case')
plt.ylabel('Time (seconds)')
plt.xticks(x, df['case'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("./times.pdf")
# plt.savefig('benchmark_fjsp/times.png')

# %%
