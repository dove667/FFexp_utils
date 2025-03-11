import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# 设置中文字体
font = set_chinese_font()

# 示例1：基本统计分析与直方图
print("\n示例1：基本统计分析与直方图")
print("-" * 50)

# 示例数据 - 某物理量的多次测量结果
data1 = [3.97, 4.06, 4.01, 3.94, 4.00, 4.06, 3.94, 4.03, 3.94, 4.06,
        3.97, 4.09, 4.06, 4.09, 4.06, 3.91, 3.94, 4.06, 4.06, 4.13]

# 计算基本统计量
stats = calculate_statistics(data1)
print("基本统计量:")
for key, value in stats.items():
    print(f"{key}: {value}")

# 计算置信区间
confidence_intervals = calculate_confidence_intervals(stats['mean'], stats['std'])
print("\n置信区间:")
for key, value in confidence_intervals.items():
    print(f"{key}: [{value[0]}, {value[1]}], 概率: {value[2]}")

# 计算sigma区间概率
sigma_probs = calculate_sigma_probabilities(data1, stats['mean'], stats['std'])
print("\nSigma区间概率:")
for key, value in sigma_probs.items():
    print(f"{key}: 理论概率: {value[0]}, 实际概率: {value[1]}")

# 绘制频率分布直方图与正态分布拟合曲线
fig, bins, n = plot_histogram_with_normal_fit(data1, x_label='测量值', 
                                           title='频率分布直方图与正态分布拟合')
plt.savefig('示例1_直方图.png', bbox_inches='tight')

# 创建统计信息表格
stat_table = create_statistics_table(stats)
print("\n统计信息表格:")
print(stat_table)

# 绘制统计信息表格
fig_stat, ax_stat = plot_table(stat_table, figsize=(8, 2), 
                             title='基本统计量', save_path='示例1_统计表.png')

# 自动划分区间并计算区间统计量
intervals = auto_create_intervals(data1, num_intervals=10)
bin_counts, bin_probs, bin_densities, total_density = calculate_interval_statistics(data1, intervals)

# 创建概率密度表格
density_table = create_density_table(intervals, bin_counts, bin_probs, bin_densities)
print("\n概率密度表格:")
print(density_table)

# 绘制概率密度表格
fig_density, ax_density = plot_table(density_table, figsize=(8, 6), 
                                   title='概率密度表', save_path='示例1_密度表.png')

# 示例2：线性回归分析
print("\n\n示例2：线性回归分析")
print("-" * 50)

# 示例数据 - 某物理实验的自变量和因变量
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y = np.array([2.1, 3.9, 6.2, 7.8, 9.9, 12.1, 14.0, 16.2, 17.9, 19.8])

# 进行线性回归分析
reg_results = linear_regression(x, y)
print("线性回归分析结果:")
for key, value in reg_results.items():
    if key != 'prediction':
        print(f"{key}: {value}")

# 使用回归方程进行预测
x_new = 5.5
y_pred = reg_results['prediction'](x_new)
print(f"\n当x={x_new}时，预测y值为: {y_pred:.3f}")

# 绘制线性回归图
fig_reg, ax_reg, _ = plot_linear_regression(x, y, x_label='X变量', y_label='Y变量', 
                                         title='线性回归分析', save_path='示例2_线性回归.png')

# 示例3：误差传递计算
print("\n\n示例3：误差传递计算")
print("-" * 50)

# 示例：计算矩形面积及其误差
# 长度和宽度及其测量误差
length = 5.0  # 单位：cm
width = 3.0   # 单位：cm
length_error = 0.1  # 单位：cm
width_error = 0.05  # 单位：cm

# 定义计算面积的函数
def area_func(l, w):
    "l * w"  # 这里的字符串会被error_propagation函数用作函数表达式
    return l * w

# 计算面积及其误差
area, area_error = error_propagation(area_func, [length, width], [length_error, width_error], ['l', 'w'])

print(f"矩形尺寸: 长 = {length} ± {length_error} cm, 宽 = {width} ± {width_error} cm")
print(f"计算面积: {area} ± {area_error} cm²")
print(f"相对误差: {(area_error/area*100):.2f}%")

# 示例4：数据导出
print("\n\n示例4：数据导出")
print("-" * 50)

# 创建示例数据
data_dict = {
    '测量值': data1,
    '预测值': [reg_results['prediction'](val) for val in data1],
    '误差': [abs(reg_results['prediction'](val) - val) for val in data1]
}

# 导出为CSV文件
export_success = export_data(data_dict, '示例4_数据导出.csv')
print(f"CSV导出{'成功' if export_success else '失败'}")

# 导出为Excel文件
export_success = export_data(data_dict, '示例4_数据导出.xlsx', format='excel')
print(f"Excel导出{'成功' if export_success else '失败'}")

# 示例5：批量数据处理
print("\n\n示例5：批量数据处理")
print("-" * 50)

# 创建多组数据
data_groups = [
    np.random.normal(10, 1, 50),  # 均值10，标准差1的正态分布
    np.random.normal(15, 2, 50),  # 均值15，标准差2的正态分布
    np.random.normal(20, 1.5, 50)  # 均值20，标准差1.5的正态分布
]

# 批量计算统计量
stats_results = batch_process(data_groups, calculate_statistics)

# 显示结果
for i, stats in enumerate(stats_results):
    print(f"\n数据组 {i+1} 的统计结果:")
    for key, value in stats.items():
        print(f"{key}: {value}")

print("\n所有示例运行完毕，请查看生成的图表和数据文件。")
plt.show()  # 显示所有图表