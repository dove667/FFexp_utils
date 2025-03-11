# 基物实验数据处理包
这是一个用于基础物理实验数据处理的Python工具包，提供了一系列函数用于数据统计分析、数据可视化和不确定度计算等功能。
由于作者能力有限，可能会存在一些错误或不足之处。如果您在使用过程中遇到任何问题，请随时与作者联系。

免责声明：本包旨在为学生提供便利，减少用在数据处理分析上的时间。对于使用此工具包带来的任何负面影响，作者不承担任何责任。
本包由作者在25春的基物实验课程的学习中开发，仅供学习交流使用。功能尚未完善，请见谅。

使用门槛：会从浏览器下载软件，不害怕在命令行复制粘贴，可以看完下面的教程。如果有python基础使用起来会更简单。
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

## 保姆级教程


### 1. 安装必要软件

#### 1.1 安装Python
1. 访问Python官网 https://www.python.org/downloads/ 下载Python（3.10左右都可以）
2. 运行安装程序，确保勾选「Add Python to PATH」选项
3. 完成安装后，打开命令提示符（按Win+R，输入cmd），输入`python --version`验证安装

#### 1.2 安装VSCode
1. 访问Visual Studio Code官网 https://code.visualstudio.com/ 下载VSCode
2. 运行安装程序，按照默认选项完成安装

### 2. 从GitHub下载代码
以下方法二选一，没有编程经验的同学选择第一种。
#### 2.1 使用浏览器下载（简单方法）
1. 访问项目GitHub页面
2. 点击绿色的「Code」按钮，然后点击「Download ZIP」
3. 解压下载的ZIP文件到你想要的位置

#### 2.2 使用Git克隆（推荐方法）
不多介绍

### 3. 安装依赖库（不会用命令行直接跳到下一步）

1. 打开命令行/终端
2. 运行以下命令安装所需的Python库：
   ```
   pip install numpy pandas matplotlib scipy sympy jupyter
   ```
3. （可选）建议使用虚拟环境，没有编程经验的同学可以跳过这一步。
### 4. 使用vscode
1. 打开vscode，open folder，选择刚刚从github下载的文件夹，打开`基物实验数据处理示例.ipynb`
2. （如果没有安装依赖去）按照vscode的报错指引一键安装缺失的依赖库
3. notebook（就是那个ipynb文件）使用方法：shift+enter运行代码或显示文字，双击编辑代码或文字，esc退出编辑模式


### 不确定度计算
特此声明，本包中的不确定度计算功能不能替代不确定度理论学习。一方面作者不保证程序计算的不确定度完全正确，最好是自己算一遍跟程序计算结果比较。另一方面作者不想看见学校的不确定度理论教学成为摆设。