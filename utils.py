import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import norm, gaussian_kde
import numpy as np
import sympy as sp
# 设置中文字体
def set_chinese_font():
    """
    设置matplotlib的中文字体显示
    
    返回:
        font: FontProperties对象，可用于设置图表中的中文字体
    """
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix' 
    return FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

# 数据统计分析函数
def calculate_statistics(data):
    """
    计算数据的基本统计量
    
    参数:
        data: 列表或numpy数组，包含需要分析的数据
        
    返回:
        dict: 包含以下统计量的字典
            - count: 数据个数
            - mean: 平均值
            - std: 标准差
            - max_value: 最大值
            - min_value: 最小值
            - median: 中位数
            - range_value: 数据范围（最大值-最小值）
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['Value'])
    
    # 计算统计量
    stats = {
        'count': round(df['Value'].count(),3),
        'mean': round(df['Value'].mean(), 3),
        'std': round(df['Value'].std(), 3),
        'max_value': round(df['Value'].max(), 3),
        'min_value': round(df['Value'].min(), 3),
        'median': round(df['Value'].median(), 3),
        'range_value': round(df['Value'].max() - df['Value'].min(), 3)
    }
    
    return stats

# 置信区间计算函数
def calculate_confidence_intervals(mean, std, confidence_levels=[0.68, 0.95, 0.99]):
    """
    计算给定置信水平的置信区间
    
    参数:
        mean: 数据平均值
        std: 数据标准差
        confidence_levels: 置信水平列表，默认为[0.68, 0.95, 0.99]
        
    返回:
        dict: 键为置信水平描述，值为(下界, 上界, 概率)的元组
    """
    confidence_interval_probabilities = {}
    
    for confidence in confidence_levels:
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        lower_bound = round(mean - z_score * std, 3)
        upper_bound = round(mean + z_score * std, 3)
        probability = round(norm.cdf(upper_bound, mean, std) - norm.cdf(lower_bound, mean, std), 3)
        confidence_interval_probabilities[f"{confidence*100}% 置信区间"] = (lower_bound, upper_bound, probability)
    
    return confidence_interval_probabilities

# sigma区间概率计算函数
def calculate_sigma_probabilities(data, mean, std):
    """
    计算数据落在σ、2σ、3σ区间内的理论概率和实际概率
    
    参数:
        data: 列表或numpy数组，包含需要分析的数据
        mean: 数据平均值
        std: 数据标准差
        
    返回:
        dict: 键为sigma描述，值为(理论概率, 实际概率)的元组
    """
    df = pd.DataFrame(data, columns=['Value'])
    sigma_intervals = [1, 2, 3]
    sigma_probabilities = {}
    
    for sigma in sigma_intervals:
        lower_bound = mean - sigma * std
        upper_bound = mean + sigma * std
        theoretical_probability = round(norm.cdf(upper_bound, mean, std) - norm.cdf(lower_bound, mean, std), 3)
        empirical_probability = round(((df['Value'] >= lower_bound) & (df['Value'] <= upper_bound)).mean(), 3)
        sigma_probabilities[f"{sigma}σ"] = (theoretical_probability, empirical_probability)
    
    return sigma_probabilities

# 自动划分区间函数
def auto_create_intervals(data, interval_width=None, num_intervals=None):
    """
    自动划分数据区间
    
    参数:
        data: 列表或numpy数组，包含需要分析的数据
        interval_width: 区间宽度，默认为None（自动计算）
        num_intervals: 区间数量，默认为None（自动计算）
        
    返回:
        numpy.ndarray: 区间边界列表
    """
    min_value = min(data)
    max_value = max(data)
    range_value = max_value - min_value
    
    # 如果没有指定区间宽度和区间数量，则自动计算
    if interval_width is None and num_intervals is None:
        # 使用Sturges规则确定区间数量
        num_intervals = int(np.ceil(1 + 3.322 * np.log10(len(data))))
        interval_width = range_value / num_intervals
    # 如果指定了区间宽度，则计算区间数量
    elif interval_width is not None:
        num_intervals = int(np.ceil(range_value / interval_width))
    # 如果指定了区间数量，则计算区间宽度
    elif num_intervals is not None:
        interval_width = range_value / num_intervals
    
    # 创建区间边界
    intervals = np.linspace(min_value, max_value, num_intervals + 1)
    
    return intervals

# 频率分布直方图与正态分布拟合函数
def plot_histogram_with_normal_fit(data, x_label='数值', title='频率分布直方图与正态分布拟合曲线', bins=15):
    """
    绘制频率分布直方图与正态分布拟合曲线
    
    参数:
        data: 列表或numpy数组，包含需要分析的数据
        x_label: x轴标签，默认为'数值'
        title: 图表标题，默认为'频率分布直方图与正态分布拟合曲线'
        bins: 直方图的箱数，默认为15
        
    返回:
        tuple: (plt.figure对象, 直方图的bins, 直方图的频率值n)
    """
    # 设置中文字体
    font = set_chinese_font()
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['Value'])
    
    # 计算统计量
    stats = calculate_statistics(data)
    mean = stats['mean']
    std = stats['std']
    
    # 创建图形
    fig = plt.figure(figsize=(10, 6), dpi=100)
    
    # 绘制频率分布直方图
    n, bins, patches = plt.hist(df['Value'], bins=bins, color='skyblue', 
                               edgecolor='black', density=True, alpha=0.6)
    
    # 标注每个频率
    for patch, frequency in zip(patches, n):
        if frequency > 0:
            plt.text(patch.get_x() + patch.get_width() / 2., patch.get_height(),
                     f'{frequency:.3f}', ha='center', va='bottom', fontproperties=font)
    
    # 计算正态分布曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label=r'Fit: $\mu$={:.3f}, $\sigma$={:.3f}'.format(mean, std))
    
    # 计算区间中值并代入正态分布函数
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_centers_pdf = norm.pdf(bin_centers, mean, std)
    
    # 在图中绘制这些点
    plt.scatter(bin_centers, bin_centers_pdf, color='red', zorder=5, label='Bin Centers')
    
    # 计算核密度估计
    kde = gaussian_kde(df['Value'])
    plt.plot(x, kde(x), 'r--', linewidth=2, label='KDE Fit')
    
    # 设置图表属性
    plt.xlabel(x_label, fontproperties=font)
    plt.ylabel('概率密度', fontproperties=font)
    plt.title(title, fontproperties=font)
    plt.legend()
    plt.grid(True)
    
    return fig, bins, n

# 计算区间频数、概率和概率密度
def calculate_interval_statistics(data, intervals):
    """
    计算数据在给定区间内的频数、概率和概率密度
    
    参数:
        data: 列表或numpy数组，包含需要分析的数据
        intervals: 区间边界列表
        
    返回:
        tuple: (bin_counts, bin_probabilities, bin_densities, total_density)
            - bin_counts: 每个区间的频数
            - bin_probabilities: 每个区间的概率
            - bin_densities: 每个区间的概率密度
            - total_density: 总概率密度
    """
    df = pd.DataFrame(data, columns=['Value'])
    
    # 计算每个区间的频数
    bin_counts = [sum((df['Value'] >= intervals[i]) & (df['Value'] < intervals[i+1])) for i in range(len(intervals)-1)]
    
    # 计算概率和概率密度
    bin_probabilities = [round(count / sum(bin_counts), 3) for count in bin_counts]
    bin_densities = [round((count / (intervals[i+1] - intervals[i])) / sum(bin_counts), 3) for i, count in enumerate(bin_counts)]
    total_density = round(sum(bin_densities), 3)
    
    return bin_counts, bin_probabilities, bin_densities, total_density

# 创建统计信息表格
def create_statistics_table(stats):
    """
    创建包含统计信息的DataFrame表格
    
    参数:
        stats: 包含统计量的字典，由calculate_statistics函数生成
        
    返回:
        DataFrame: 格式化后的统计信息表格
    """
    stat_df_table = pd.DataFrame({
        '数据个数': [stats['count']],
        '平均值': [stats['mean']],
        '标准差': [stats['std']],
        '最大值': [stats['max_value']],
        '最小值': [stats['min_value']],
        '中位数': [stats['median']],
        '范围': [stats['range_value']]
    })
    
    # 格式化统计信息表格数据到小数点后三位
    stat_df_table_formatted = stat_df_table.copy()
    for col in stat_df_table_formatted.columns:
        stat_df_table_formatted[col] = stat_df_table_formatted[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    
    return stat_df_table_formatted

# 创建概率密度表格
def create_density_table(intervals, bin_counts, bin_probabilities, bin_densities, total_density=None):
    """
    创建包含区间、频数、概率和概率密度的DataFrame表格
    
    参数:
        intervals: 区间边界列表
        bin_counts: 每个区间的频数
        bin_probabilities: 每个区间的概率
        bin_densities: 每个区间的概率密度
        total_density: 总概率密度，默认为None（自动计算）
        
    返回:
        DataFrame: 格式化后的概率密度表格
    """
    # 创建区间标签
    interval_labels = [f"{intervals[i]:.3f}-{intervals[i+1]:.3f}" for i in range(len(intervals)-1)]
    
    # 如果没有提供总概率密度，则计算它
    if total_density is None:
        total_density = round(sum(bin_densities), 3)
    
    # 创建概率密度表格
    density_df_table = pd.DataFrame({
        '区间': interval_labels + ['合计'],
        '频数': bin_counts + [sum(bin_counts)],
        '概率': bin_probabilities + [round(sum(bin_probabilities), 3)],
        '概率密度': bin_densities + [total_density]
    })
    
    # 格式化概率密度表格数据到小数点后三位
    # 使用map替代已弃用的applymap
    density_df_table_formatted = density_df_table.copy()
    for col in density_df_table_formatted.columns:
        if col != '区间':  # 区间列不需要格式化
            density_df_table_formatted[col] = density_df_table_formatted[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    
    return density_df_table_formatted

# 绘制表格图像
# 绘制表格图像
def plot_table(df_table, figsize=(6, 4), title=None, save_path=None):
    """
    将DataFrame表格绘制为图像
    
    参数:
        df_table: DataFrame表格
        figsize: 图像大小，默认为(6, 4)
        title: 表格标题，默认为None
        save_path: 保存路径，默认为None（不保存）
        
    返回:
        tuple: (fig, ax) matplotlib的图形和坐标轴对象
    """
    # 设置中文字体
    font = set_chinese_font()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')  # 不显示坐标轴
    
    # 绘制表格
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, 
                     cellLoc='center', loc='center')
    
    # 设置表格标题
    if title is not None:
        plt.title(title, fontproperties=font)
    
    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # 调整表格大小
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax

# 线性回归分析函数
def linear_regression(x, y):
    """
    对数据进行线性回归分析
    
    参数:
        x: 自变量数据，列表或numpy数组
        y: 因变量数据，列表或numpy数组
        
    返回:
        dict: 包含以下回归分析结果的字典
            - slope: 斜率
            - intercept: 截距
            - r_squared: 决定系数R²
            - std_error: 标准误差
            - prediction: 预测函数
    """
    # 将数据转换为numpy数组
    x = np.array(x)
    y = np.array(y)
    
    # 计算线性回归参数
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率和截距
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = round(numerator / denominator, 3)
    intercept = round(y_mean - slope * x_mean, 3)
    
    # 计算预测值
    y_pred = slope * x + intercept
    
    # 计算决定系数R²
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = round(1 - (ss_residual / ss_total), 3)
    
    # 计算标准误差
    std_error = round(np.sqrt(ss_residual / (n - 2)), 3)
    
    # 创建预测函数
    def prediction(x_new):
        return slope * x_new + intercept
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'std_error': std_error,
        'prediction': prediction
    }

# 计算A类不确定度
def calculate_uncertainty_a(data):
    """
    计算A类不确定度（统计分析得到的标准偏差）
    
    参数:
        data: 列表或numpy数组，包含多次测量的数据
        
    返回:
        float: A类不确定度（标准偏差）
    """
    data_array = np.array(data)
    n = len(data_array)
    mean = np.mean(data_array)
    # 计算实验标准差
    std_dev = np.sqrt(np.sum((data_array - mean) ** 2) / (n - 1))
    # 计算平均值的标准不确定度（A类不确定度）
    mu_a = std_dev / np.sqrt(n)
    return round(mu_a, 3)

# 计算B类不确定度
def calculate_uncertainty_b(estimation, resolution, distribution='uniform'):
    """
    计算B类不确定度（仪器精度等系统误差）
    
    参数:
        estimation: 估计不确定度
        resolution: 仪器不确定度
        distribution: 分布类型，可选'uniform'（均匀分布）或'normal'（正态分布），默认为'uniform'
        
    返回:
        float: B类不确定度
    """
    if distribution.lower() == 'uniform':
        # 均匀分布，除以根号3
        mu_b = np.sqrt(estimation**2 + resolution**2) / np.sqrt(3)
    elif distribution.lower() == 'normal':
        # 正态分布，除以3(默认置信概率为1)
        mu_b = np.sqrt(estimation**2 + resolution**2) / 3
    else:
        raise ValueError("分布类型必须是'uniform'或'normal'")
    return round(mu_b, 3)

# 合成不确定度
def combine_uncertainties(t,mu_a,kp,mu_b):
    """
    合成不确定度（平方和开方法）
    
    参数:
        t，kp: t因子，包含因子 (自行查表)
        mu_a,mu_B: A,B类不确定度
        
    返回:
        float: 合成不确定度
    """
    # 平方和开方法计算合成不确定度
    u_c = np.sqrt((t*mu_a)**2 + (kp*mu_b)**2)
    return round(u_c, 3)
# 不确定度传递
def uncertainty_propagation(func, values, uncertainties, variables=None):
    """
    使用不确定度传递公式计算复合函数的不确定度
    
    参数:
        func: 计算函数，接受与values长度相同的参数
        values: 变量的值列表
        uncertainties: 变量的不确定度列表
        variables: 变量名列表，默认为None（自动生成）
        
    返回:
        tuple: (result, uncertainty) 计算结果和不确定度
    """
    # 如果没有提供变量名，则自动生成
    if variables is None:
        variables = [f'x{i}' for i in range(len(values))]
    
    # 创建符号变量
    sym_vars = sp.symbols(variables)
    
    # 将函数转换为符号表达式
    if callable(func):
        # 创建lambda函数的字符串表示
        args_str = ', '.join(variables)
        func_str = f"lambda {args_str}: {func.__doc__.strip()}"
        # 使用eval将字符串转换为lambda函数
        sym_func = eval(func_str)(*sym_vars)
    else:
        # 如果func已经是字符串表达式
        sym_func = sp.sympify(func)
    
    # 计算每个变量的偏导数
    derivatives = [sp.diff(sym_func, var) for var in sym_vars]
    
    # 将值代入偏导数表达式
    deriv_values = []
    for deriv in derivatives:
        deriv_func = sp.lambdify(sym_vars, deriv)
        deriv_values.append(deriv_func(*values))
    
    # 计算不确定度传递（平方和开方法）
    squared_uncertainty_terms = [(deriv_values[i] * uncertainties[i]) ** 2 for i in range(len(values))]
    total_uncertainty = np.sqrt(sum(squared_uncertainty_terms))
    
    # 计算函数值
    result_func = sp.lambdify(sym_vars, sym_func)
    result = result_func(*values)
    
    return round(result, 3), round(total_uncertainty, 3)

# 单摆测量重力加速度函数
def pendulum_gravity(L, T):
    """
    4 * (3.14159**2) * L / (T**2)
    """
    return 4 * (np.pi**2) * L / (T**2)

# 绘制线性回归图函数
def plot_linear_regression(x, y, x_label='X', y_label='Y', title='线性回归分析', save_path=None):
    """
    绘制线性回归图
    
    参数:
        x: 自变量数据，列表或numpy数组
        y: 因变量数据，列表或numpy数组
        x_label: x轴标签，默认为'X'
        y_label: y轴标签，默认为'Y'
        title: 图表标题，默认为'线性回归分析'
        save_path: 保存路径，默认为None（不保存）
        
    返回:
        tuple: (fig, ax, regression_results) matplotlib的图形、坐标轴对象和回归分析结果
    """
    # 设置中文字体
    font = set_chinese_font()
    
    # 进行线性回归分析
    reg_results = linear_regression(x, y)
    slope = reg_results['slope']
    intercept = reg_results['intercept']
    r_squared = reg_results['r_squared']
    std_error = reg_results['std_error']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 绘制散点图
    ax.scatter(x, y, color='blue', alpha=0.6, label='数据点')
    
    # 绘制回归线
    x_min, x_max = min(x), max(x)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = reg_results['prediction'](x_line)
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'拟合线: y = {slope:.3f}x + {intercept:.3f}')
    
    # 添加回归方程和R²值
    equation_text = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_squared:.3f}\n标准误差 = {std_error:.3f}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, 
            fontproperties=font, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 设置图表属性
    ax.set_xlabel(x_label, fontproperties=font)
    ax.set_ylabel(y_label, fontproperties=font)
    ax.set_title(title, fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax, reg_results

# 数据导出函数
def export_data(data, file_path, format='csv', sheet_name='Sheet1'):
    """
    将数据导出到文件
    
    参数:
        data: DataFrame或字典，包含需要导出的数据
        file_path: 导出文件路径
        format: 导出格式，支持'csv'、'excel'、'txt'，默认为'csv'
        sheet_name: Excel工作表名称，仅在format='excel'时有效，默认为'Sheet1'
        
    返回:
        bool: 导出是否成功
    """
    try:
        # 确保data是DataFrame
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            else:
                data = pd.DataFrame(data)
        
        # 根据格式导出数据
        if format.lower() == 'csv':
            data.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif format.lower() == 'excel':
            data.to_excel(file_path, sheet_name=sheet_name, index=False)
        elif format.lower() == 'txt':
            data.to_csv(file_path, sep='\t', index=False, encoding='utf-8-sig')
        else:
            print(f"不支持的导出格式: {format}")
            return False
        
        print(f"数据已成功导出到: {file_path}")
        return True
    except Exception as e:
        print(f"导出数据时出错: {e}")
        return False

# 批量数据处理函数
def batch_process(data_list, process_func, *args, **kwargs):
    """
    批量处理多组数据
    
    参数:
        data_list: 数据列表，每个元素是一组需要处理的数据
        process_func: 处理函数，接受一组数据和其他参数
        *args, **kwargs: 传递给处理函数的其他参数
        
    返回:
        list: 处理结果列表
    """
    results = []
    for i, data in enumerate(data_list):
        try:
            result = process_func(data, *args, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"处理第{i+1}组数据时出错: {e}")
            results.append(None)
    
    return results