# config.py
import matplotlib.pyplot as plt

class ProjectConfig:
    """项目配置"""
    # 特征选择
    FEATURES_2D = [2, 3]
    FEATURES_3D = [1, 2, 3]
    FIXED_FEATURE_IDX = 0
    
    # 实验参数
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    CV_FOLDS = 5

    # === 可视化参数 ===
    # 分辨率
    RESOLUTION_2D = 200    # 2D网格分辨率
    RESOLUTION_3D = 50     # 3D网格分辨率
    
    # 颜色方案
    COLORS = {
        'setosa': '#FF6B6B',
        'versicolor': '#4ECDC4',
        'virginica': '#45B7D1',
        'background': ['#FFCCCC', '#CCFFCC', '#CCCCFF'],
        'grid': '#E0E0E0',
        'boundary': '#333333',
        'text': '#2C3E50'
    }
    
    # 输出设置
    SAVE_FIGURES = True
    OUTPUT_DIR = 'results/'
    FIGURE_DPI = 300
    
    @classmethod
    def setup_style(cls):
        """设置matplotlib样式"""
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn')
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlepad'] = 15
        plt.rcParams['axes.labelpad'] = 10