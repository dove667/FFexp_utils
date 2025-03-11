import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import norm, gaussian_kde
import numpy as np

# 您的数据
data = [87.85, 87.75, 88.15, 87.60, 87.63]


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

# 输出统计量
print(f"数据个数: {count}")
print(f"平均值: {mean}")
print(f"标准差: {std}")
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
print(f"中间值（中位数）: {median}")
print(f"范围: {range_value}")

# 计算置信区间
confidence_intervals = [0.68, 0.95, 0.99]  # 修改置信区间
confidence_interval_probabilities = {}

for confidence in confidence_intervals:
    z_score = norm.ppf(1 - (1 - confidence) / 2)
    lower_bound = mean - z_score * std
    upper_bound = mean + z_score * std
    probability = norm.cdf(upper_bound, mean, std) - norm.cdf(lower_bound, mean, std)
    confidence_interval_probabilities[f"{confidence*100}% 置信区间"] = (lower_bound, upper_bound, probability)

# 输出置信区间及其概率
for key, value in confidence_interval_probabilities.items():
    print(f"{key}: [{value[0]:.3f}, {value[1]:.3f}]，概率: {value[2]:.3f}")

# 计算落在σ、2σ、3σ之间的概率
sigma_intervals = [1, 2, 3]
sigma_probabilities = {}

for sigma in sigma_intervals:
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    probability = norm.cdf(upper_bound, mean, std) - norm.cdf(lower_bound, mean, std)
    empirical_probability = ((df['Value'] >= lower_bound) & (df['Value'] <= upper_bound)).mean()
    sigma_probabilities[f"{sigma}σ"] = (probability, empirical_probability)

# 输出σ、2σ、3σ之间的概率与理论值比较
print("\nσ、2σ、3σ之间的概率与理论值比较:")
for key, value in sigma_probabilities.items():
    print(f"{key}: 理论概率: {value[0]:.3f}, 实际概率: {value[1]:.3f}")

# 设置字体属性
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows 系统使用

# 绘制频率分布直方图
plt.figure(figsize=(10, 6), dpi=100)
# 绘制频率分布直方图，density=True 参数用于将直方图的频率转换为概率密度，使得直方图的总面积为1
n, bins, patches = plt.hist(df['Value'], bins=15, color='skyblue', edgecolor='black', density=True, alpha=0.6)

# 标注每个频率
for patch, frequency in zip(patches, n):
    if frequency > 0:
        plt.text(patch.get_x() + patch.get_width() / 2., patch.get_height(),
                 f'{frequency:.3f}', ha='center', va='bottom', fontproperties=font)

# 计算正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std) #pdf probability density function
plt.plot(x, p, 'k', linewidth=2, label=r'Fit: $\mu$={:.3f}, $\sigma$={:.3f}'.format(mean, std))

# 计算区间中值并代入正态分布函数
bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 区间中值
bin_centers_pdf = norm.pdf(bin_centers, mean, std)  # 区间中值对应的正态分布概率密度值

# 在图中绘制这些点
plt.scatter(bin_centers, bin_centers_pdf, color='red', zorder=5, label='Bin Centers')

# 计算核密度估计
kde = gaussian_kde(df['Value'])
plt.plot(x, kde(x), 'r--', linewidth=2, label='KDE Fit')

plt.xlabel('时间/s', fontproperties=font)
plt.ylabel('概率密度', fontproperties=font)
plt.title('频率分布直方图与正态分布拟合曲线', fontproperties=font)
plt.legend()
plt.grid(True)
plt.show()

# 统计信息表格
stat_df_table = pd.DataFrame({
    '数据个数': [count],
    '平均值': [mean],
    '标准差': [std],
    '最大值': [max_value],
    '最小值': [min_value],
    '中间值（中位数）': [median],
    '范围': [range_value]
})

# 格式化统计信息表格数据到小数点后三位
stat_df_table_formatted = stat_df_table.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

# 计算每个区间的概率和概率密度
intervals = bins
bin_counts = [sum((df['Value'] >= intervals[i]) & (df['Value'] < intervals[i+1])) for i in range(len(intervals)-1)]

# 计算概率和概率密度
bin_probabilities = [round(count / sum(bin_counts), 3) for count in bin_counts]
bin_densities = [round((count / (intervals[i+1] - intervals[i])) / sum(bin_counts), 3) for i, count in enumerate(bin_counts)]
total_density = round(sum(bin_densities), 3)  # 合计概率密度

# 概率密度表格
density_df_table = pd.DataFrame({
    '区间': [f"{intervals[i]:.3f}-{intervals[i+1]:.3f}" for i in range(len(intervals)-1)],
    '频数': bin_counts,
    '概率': bin_probabilities,
    '概率密度': bin_densities
})

# 格式化概率密度表格数据到小数点后三位
density_df_table_formatted = density_df_table.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)