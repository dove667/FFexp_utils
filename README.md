# 基物实验数据处理包

这是一个用于基础物理实验数据处理的Python工具包，提供了一系列函数用于数据统计分析、可视化和不确定度计算等功能。

## 主要功能

### 1. 数据统计分析
- 基本统计量计算（均值、标准差、中位数等）
- 置信区间计算
- Sigma区间概率计算
- 区间频数、概率和概率密度计算
- 自动划分数据区间

### 2. 数据可视化
- 频率分布直方图与正态分布拟合
- 统计信息表格绘制
- 概率密度表格绘制
- 线性回归图绘制
- 表格数据可视化

### 3. 数据处理
- 线性回归分析
- 不确定度传递计算
- 数据导出（CSV、Excel、TXT）
- 批量数据处理
- A类不确定度计算
- B类不确定度计算
- 合成不确定度计算
- 间接不确定度的传递

## 从零开始使用指南

### 1. 安装必要软件

#### 1.1 安装Python
1. 访问Python官网 https://www.python.org/downloads/ 下载最新版Python
2. 运行安装程序，确保勾选「Add Python to PATH」选项
3. 完成安装后，打开命令提示符（按Win+R，输入cmd），输入`python --version`验证安装

#### 1.2 安装VSCode
1. 访问Visual Studio Code官网 https://code.visualstudio.com/ 下载VSCode
2. 运行安装程序，按照默认选项完成安装
3. 打开VSCode，安装Python扩展：点击左侧扩展图标，搜索「Python」并安装

### 2. 从GitHub下载代码

#### 2.1 使用浏览器下载（简单方法）
1. 访问项目GitHub页面
2. 点击绿色的「Code」按钮，然后点击「Download ZIP」
3. 解压下载的ZIP文件到你想要的位置

#### 2.2 使用Git克隆（推荐方法）
1. 安装Git：访问 https://git-scm.com/downloads 下载并安装Git
2. 打开命令提示符，使用cd命令导航到你想保存代码的文件夹
3. 输入`git clone [项目GitHub地址]`命令克隆仓库

### 3. 安装依赖库

1. 打开命令行/终端
2. 运行以下命令安装所需的Python库：
   ```
   pip install numpy pandas matplotlib scipy sympy jupyter
   ```
3. （可选）建议使用虚拟环境，如果不需要管理不同项目的依赖配置，可以跳过这一步。
### 4. 使用Jupyter Notebook
1. vscode下载jupyter插件
2. 使用方法：shift+enter运行代码或显示markdown，双击编辑代码或markdown，esc退出编辑模式
## 使用方法详见```基物实验数据处理示例.ipynb```

### 不确定度计算
未开发完毕,详情见
`基物实验数据处理示例.ipynb`

## 依赖库

- numpy
- pandas
- matplotlib
- scipy
- sympy 
- jupyter 

## 注意事项

- 所有数值输出均格式化到小数点后三位
- 首次使用时请先运行示例文件熟悉各函数的用法