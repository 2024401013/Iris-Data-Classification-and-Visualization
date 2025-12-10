# 🌸 Iris Data Classification and Visualization Project

一个完整的鸢尾花（Iris）数据集分类与可视化项目，涵盖多种机器学习分类器的比较、2D/3D决策边界可视化、概率热图等高级功能。

## 📋 项目概述

本项目基于经典的鸢尾花数据集，实现了完整的机器学习工作流：
- **数据探索**：特征分布、相关性分析
- **多分类器比较**：9种主流机器学习算法
- **2D可视化**：决策边界、概率热图
- **3D可视化**：概率曲面、决策边界
- **交互式可视化**：Plotly支持

## 🚀 快速开始

### 环境要求

```bash
# 基础依赖
pip install numpy pandas matplotlib seaborn scikit-learn

# 3D可视化依赖
pip install scikit-image

# 可选：交互式可视化
pip install plotly
```

### 项目结构

```
iris-project/
├── main.py                      # 主程序入口
├── config.py                    # 配置管理
├── data_manager.py              # 数据加载与预处理
├── classifier_evaluator.py      # 分类器训练与评估
├── utils.py                     # 工具函数
└── visualization/               # 可视化模块
    ├── __init__.py
    ├── base_visualizer.py       # 基础可视化
    ├── task1_2d_visualizer.py   # 任务1：2D决策边界
    ├── task23_3d_visualizer.py  # 任务2/3：3D概率曲面
    ├── task4_3d_combined.py     # 任务4：3D边界+概率图
    └── performance_visualizer.py # 性能对比
```

### 运行项目

```bash
# 直接运行主程序
python main.py

# 或指定输出目录
python main.py --output results/
```

## 📊 任务清单

### ✅ 任务1：2D分类边界与概率图
- **特征选择**：三分类 + 两个特征
- **分类器比较**：9种机器学习算法
- **可视化**：
  - 决策边界对比图
  - 概率热图（不确定性可视化）
  - 分类器性能排名

### ✅ 任务2：3D决策边界
- **特征选择**：二分类 + 三个特征
- **可视化**：3D决策边界曲面
- **技术**：固定一个维度，显示概率分布

### ✅ 任务3：3D概率图
- **特征选择**：二分类 + 三个特征
- **可视化**：3D概率曲面
- **技术**：使用逻辑回归生成平滑概率分布

### ✅ 任务4：3D边界 + 概率图
- **特征选择**：三分类 + 三个特征
- **可视化**：
  - 左图：三对类别间的决策边界
  - 右图：最大概率热力图
  - 补充：2D投影视图

## 🔧 配置说明

在 `config.py` 中可以修改以下参数：

```python
# 特征选择
FEATURES_2D = [2, 3]           # 2D可视化使用的特征索引
FEATURES_3D = [1, 2, 3]        # 3D可视化使用的特征索引

# 实验参数
TEST_SIZE = 0.3                # 测试集比例
RANDOM_STATE = 42              # 随机种子
CV_FOLDS = 5                   # 交叉验证折数

# 可视化参数
RESOLUTION_2D = 200            # 2D网格分辨率
RESOLUTION_3D = 50             # 3D网格分辨率
SAVE_FIGURES = True            # 是否保存图像
OUTPUT_DIR = 'results/'        # 输出目录
```

## 📈 支持的分类器

项目实现了9种分类器的对比：

| 分类器 | 算法类型 | 支持概率预测 |
|--------|----------|--------------|
| Logistic Regression | 线性模型 | ✅ |
| SVM (Linear) | 支持向量机 | ✅ |
| SVM (RBF) | 支持向量机 | ✅ |
| Decision Tree | 树模型 | ✅ |
| Random Forest | 集成学习 | ✅ |
| Gradient Boosting | 集成学习 | ✅ |
| AdaBoost | 集成学习 | ✅ |
| Naive Bayes | 概率模型 | ✅ |
| K-NN (k=5) | 近邻算法 | ✅ |

## 🎨 可视化特色

### 配色方案
- **setosa**: #FF6B6B (珊瑚红)
- **versicolor**: #4ECDC4 (青绿色)
- **virginica**: #45B7D1 (天空蓝)

### 输出图像
运行后将在 `results/` 目录下生成：

1. **特征分析**
   - `feature_distribution.png` - 特征分布（小提琴图+箱线图）
   - `correlation_analysis.png` - 特征相关性矩阵

2. **2D可视化**
   - `2d_decision_boundaries.png` - 9分类器决策边界对比
   - `probability_maps_*.png` - 各分类器概率热图

3. **3D可视化**
   - `3d_probability_surface.png` - 3D概率曲面（二分类）
   - `task4_3d_boundary_probability.png` - 3D边界+概率图（三分类）
   - `task4_2d_projections.png` - 2D投影视图

4. **性能分析**
   - `performance_comparison.png` - 分类器综合性能对比

## 📊 技术特点

### 1. 模块化设计
- 数据管理、分类器评估、可视化分离
- 每个任务独立模块，易于维护和扩展
- 清晰的接口和职责分离

### 2. 健壮性
- 完整的异常处理
- 配置检查与验证
- 优雅降级（如scikit-image不可用）

### 3. 可扩展性
- 易于添加新分类器
- 支持自定义特征组合
- 可调整可视化参数

### 4. 实用性
- 自动创建输出目录
- 高DPI图像保存
- 详细的运行日志

## 🔍 关键发现

基于本项目分析，我们可以得出以下结论：

1. **最佳特征**：花瓣长度和宽度提供最佳的类别分离
2. **最佳分类器**：集成学习方法（Random Forest, Gradient Boosting）在鸢尾花数据集上表现优异
3. **可视化价值**：3D可视化能清晰展示概率在特征空间中的连续分布
4. **决策边界**：概率=0.5的等值面能有效区分不同类别

## 🛠️ 故障排除

### 常见问题

1. **scikit-image导入错误**
   ```bash
   # 安装正确版本
   pip install scikit-image==0.19.0
   ```

2. **内存不足**
   - 降低 `RESOLUTION_3D` 值（默认为50）
   - 减少采样点数量

3. **中文乱码**
   - 项目已设置为英文显示
   - 如需中文，请修改matplotlib字体配置

4. **图像保存失败**
   - 检查输出目录权限
   - 确保磁盘空间充足

### 调试建议

```python
# 在main.py开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```
