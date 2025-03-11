import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体属性，确保中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 您的数据
data = [3.97,4.06,4.01,3.94,4.00,4.06,3.94,4.03,3.94,4.06,
        3.97,4.09,4.06,4.09,4.06,3.91,3.94,4.06,4.06,4.13,
        4.00,4.00,4.06,3.97,4.03,4.06,3.94,4.06,4.00,4.06,
        4.03,4.06,4.03,4.02,4.00,4.18,4.00,3.97,4.03,4.12,
        4.09,4.00,4.06,4.03,4.08,4.00,4.12,4.06,3.98,3.97,
        4.03,4.00,4.03,3.97,4.03,3.93,3.88,4.00,3.81,4.06,
        4.06,4.03,3.94,4.03,4.09,4.06,4.00,3.97,3.87,3.97,
        3.97,4.18,4.03,3.97,4.03,4.06,4.03,4.00,3.87,4.06,
        4.00,4.00,3.97,4.03,4.09,3.93,3.06,4.06,3.96,4.00,
        4.03,4.00,4.00,3.94,3.97,3.97,3.97,4.10,3.93,4.03,
        4.03,4.06,4.00,4.07,4.00,4.03,4.00,4.06,3.94,3.94,
        3.94,3.88,3.94,3.91,3.94,4.04,3.93,4.07,3.94,4.00,
        3.94,4.06,4.00,3.94,4.00,3.97,3.90,3.97,4.00,4.00,
        4.09,4.03,4.09,4.03,4.03,4.00,3.81,4.06,4.06,3.90,
        4.13,3.94,4.09,3.97,4.12,4.03,3.93,4.03,3.93,4.03,
        4.00,4.15,3.81,3.91,3.97,4.09,4.00,3.94,3.97,3.94,
        4.00,3.94,3.93,4.00,3.96,3.90,4.00,3.93,3.94,3.96,
        4.00,3.91,3.96,3.76,4.07,4.07,3.96,4.00,4.00,4.00,
        4.03,3.97,4.00,4.04,4.07,3.94,3.93,3.93,3.90,4.00,
        4.00,3.75,4.10,4.00,4.07,3.90,4.10,4.00,4.10,4.03
]

# 将数据转换为 Pandas DataFrame
df = pd.DataFrame(data, columns=['Value'])

# 计算统计量
count = df['Value'].count()
mean = round(df['Value'].mean(), 3)
std = round(df['Value'].std(), 3)
max_value = round(df['Value'].max(), 3)
min_value = round(df['Value'].min(), 3)
median = round(df['Value'].median(), 3)
range_value = round(df['Value'].max() - df['Value'].min(), 3)

# 创建统计信息表格数据
stat_table_data = {
    '数据个数': [count],
    '平均值': [mean],
    '标准差': [std],
    '最大值': [max_value],
    '中间值': [median],
    '最小值': [min_value],
    '范围': [range_value]
}

stat_df_table = pd.DataFrame(stat_table_data)

# 格式化统计信息表格数据到小数点后三位
stat_df_table_formatted = stat_df_table.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

# 合理划分区间间隔Δ，统计落在每个区间的频数（数据个数），并计算概率和概率密度（有单位）
intervals = np.arange(min_value, max_value, 0.03)  # 将数据分为以0.03为间隔的区间
interval_labels = [f"[{intervals[i]:.2f}, {intervals[i+1]:.2f})" for i in range(len(intervals)-1)]  # 区间标签

# 统计落在每个区间的频数
bin_counts = []
for i in range(len(intervals) - 1):
    count = len(df[(df['Value'] >= intervals[i]) & (df['Value'] < intervals[i+1])])
    bin_counts.append(count)

# 计算概率和概率密度
bin_probabilities = [round(count / sum(bin_counts), 3) for count in bin_counts]
bin_densities = [round((count / (intervals[i+1] - intervals[i])) / sum(bin_counts), 3) for i, count in enumerate(bin_counts)]
total_density = round(sum(bin_densities), 3)  # 合计概率密度

# 创建概率密度表格数据
density_table_data = {
    '区间': interval_labels + ['合计'],
    '频数': bin_counts + [200],
    '概率': bin_probabilities + [1],
    '概率密度': bin_densities + [total_density]
}

density_df_table = pd.DataFrame(density_table_data)

# 格式化概率密度表格数据到小数点后三位
density_df_table_formatted = density_df_table.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

# 绘制统计信息表格
fig1, ax1 = plt.subplots(figsize=(6, 2))
ax1.axis('off')  # 不显示坐标轴
ax1.table(cellText=stat_df_table_formatted.values, colLabels=stat_df_table_formatted.columns, cellLoc='center', loc='center')
plt.savefig('stat_table.png', bbox_inches='tight')  # 保存统计信息表格为图片
plt.show()  # 显示统计信息表格

# 绘制概率密度表格
fig2, ax2 = plt.subplots(figsize=(6, 8))  # 增加图片高度
ax2.axis('off')  # 不显示坐标轴
ax2.table(cellText=density_df_table_formatted.values, colLabels=density_df_table_formatted.columns, cellLoc='center', loc='center')
plt.savefig('density_table.png', bbox_inches='tight')  # 保存概率密度表格为图片
plt.show()  # 显示概率密度表格