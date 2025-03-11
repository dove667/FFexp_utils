# 基物实验数据处理包

这是一个用于物理实验数据处理的Python工具包，提供了一系列函数用于数据统计分析、可视化和误差计算等功能。

## 主要功能

### 1. 数据统计分析
- 基本统计量计算（均值、标准差、中位数等）
- 置信区间计算
- Sigma区间概率计算
- 区间频数、概率和概率密度计算

### 2. 数据可视化
- 频率分布直方图与正态分布拟合
- 统计信息表格绘制
- 概率密度表格绘制
- 线性回归图绘制

### 3. 数据处理
- 线性回归分析
- 误差传递计算
- 数据导出（CSV、Excel、TXT）
- 批量数据处理

## 使用方法

### 基本统计分析
```python
from utils import calculate_statistics

# 示例数据
data = [3.97, 4.06, 4.01, 3.94, 4.00, 4.06]

# 计算基本统计量
stats = calculate_statistics(data)
print(stats)
```

### 绘制频率分布直方图
```python
from utils import plot_histogram_with_normal_fit

# 绘制频率分布直方图与正态分布拟合曲线
fig, bins, n = plot_histogram_with_normal_fit(data, x_label='测量值',  title='频率分布直方图与正态分布拟合')
```

### 线性回归分析
```python
from utils import linear_regression, plot_linear_regression

# 示例数据
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.1, 3.9, 6.2, 7.8, 9.9]

# 进行线性回归分析
reg_results = linear_regression(x, y)

# 绘制线性回归图
fig_reg, ax_reg, _ = plot_linear_regression(x, y, x_label='X变量', y_label='Y变量', title='线性回归分析')
```

### 误差传递计算
```python
from utils import error_propagation

# 定义计算函数
def area_func(l, w):
    "l * w"  # 这里的字符串会被error_propagation函数用作函数表达式
    return l * w

# 计算面积及其误差
area, area_error = error_propagation(area_func, [5.0, 3.0], [0.1, 0.05], ['l', 'w'])
```

## 示例脚本

包含了一个完整的示例脚本 `示例_数据处理.py`，展示了如何使用本工具包进行物理实验数据处理，包括：

1. 基本统计分析与直方图
2. 线性回归分析
3. 误差传递计算
4. 数据导出
5. 批量数据处理

运行示例脚本：
```
python 示例_数据处理.py
```

## 依赖库

- numpy
- pandas
- matplotlib
- scipy
- sympy (用于误差传递计算)

## 注意事项

- 所有数值输出均格式化到小数点后三位
- 图表中文显示需要安装中文字体（默认使用SimHei）
- 误差传递计算需要安装sympy库