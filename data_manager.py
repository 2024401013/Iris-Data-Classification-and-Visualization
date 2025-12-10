# data_manager.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

class IrisDataManager:
    """鸢尾花数据管理类"""
    
    def __init__(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                             'petal length (cm)', 'petal width (cm)']
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = [self.target_names[i] for i in self.y]
    
    def get_2d_data(self, features=None):
        """获取2D数据（三分类）"""
        if features is None:
            from config import ProjectConfig
            features = ProjectConfig.FEATURES_2D
        X_2d = self.X[:, features]
        return X_2d, self.y
    
    def get_3d_data(self, features=None, binary=True):
        """获取3D数据（可选二分类）"""
        if features is None:
            from config import ProjectConfig
            features = ProjectConfig.FEATURES_3D
        X_3d = self.X[:, features]
        
        if binary:
            mask = self.y < 2
            return X_3d[mask], self.y[mask]
        return X_3d, self.y
    
    def get_feature_names(self, indices):
        """根据索引获取特征名"""
        return [self.feature_names[i] for i in indices]