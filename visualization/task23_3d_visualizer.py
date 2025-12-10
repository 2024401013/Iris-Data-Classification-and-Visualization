# visualization/task23_3d_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Task23Visualizer:
    """任务2/3：3D概率曲面"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_3d_probability_surface(self, X, y, classifier, feature_names):
        """3D probability surface visualization"""
        # Data preparation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        classifier.fit(X_train_scaled, y_train)
        accuracy = classifier.score(X_test_scaled, y_test)
        
        # Create grid
        feat1, feat2 = 0, 1
        fixed_value = np.mean(X_train_scaled[:, self.config.FIXED_FEATURE_IDX])
        
        x_min, x_max = X_train_scaled[:, feat1].min() - 1, X_train_scaled[:, feat1].max() + 1
        y_min, y_max = X_train_scaled[:, feat2].min() - 1, X_train_scaled[:, feat2].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, self.config.RESOLUTION_3D),
            np.linspace(y_min, y_max, self.config.RESOLUTION_3D)
        )
        
        # Calculate probabilities
        if hasattr(classifier, 'predict_proba'):
            grid_points = np.c_[
                xx.ravel(), 
                yy.ravel(), 
                np.full_like(xx.ravel(), fixed_value)
            ]
            
            probs = classifier.predict_proba(grid_points)[:, 1]  # versicolor probability
            zz = probs.reshape(xx.shape)
        else:
            print(f"⚠️ {classifier.__class__.__name__} does not support probability prediction")
            return None
        
        # 3D Plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set view
        ax.view_init(elev=30, azim=45)
        
        # Plot background grid
        bg_x = np.linspace(x_min, x_max, 15)
        bg_y = np.linspace(y_min, y_max, 15)
        bg_xx, bg_yy = np.meshgrid(bg_x, bg_y)
        bg_zz = np.zeros_like(bg_xx)
        
        ax.plot_wireframe(bg_xx, bg_yy, bg_zz, 
                         color=self.config.COLORS['grid'],
                         alpha=0.3, linewidth=0.5)
        
        # Plot probability surface
        surf = ax.plot_surface(xx, yy, zz,
                              cmap='viridis',
                              alpha=0.85,
                              linewidth=0.1,
                              antialiased=True)
        
        # Add data points
        colors = [self.config.COLORS['setosa'], self.config.COLORS['versicolor']]
        
        for i in range(2):
            mask = y_train == i
            ax.scatter(X_train_scaled[mask, feat1],
                      X_train_scaled[mask, feat2],
                      np.zeros(np.sum(mask)) - 0.05,
                      c=colors[i], label=['setosa', 'versicolor'][i],
                      s=40, alpha=0.8, edgecolor='black')
        
        # Axis labels
        ax.set_xlabel(f'{feature_names[feat1]}\n(standardized)', fontsize=11, labelpad=12)
        ax.set_ylabel(f'{feature_names[feat2]}\n(standardized)', fontsize=11, labelpad=12)
        ax.set_zlabel('Probability\n(versicolor)', fontsize=11, labelpad=12)
        
        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Probability of versicolor', fontsize=11)
        
        # Title
        title = (f'3D Probability Surface - {classifier.__class__.__name__}\n'
                f'Accuracy: {accuracy:.3f} | Fixed feature: {feature_names[self.config.FIXED_FEATURE_IDX]}')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        ax.legend()
        plt.tight_layout()
        
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}3d_probability_surface.png", 
                       dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        
        return fig

    def plot_ppt_style_decision_plane(self, X, y, feature_names):
        """
        绘制PPT风格的线性决策平面示意图
        适用于任务2：二分类，三个特征
        """
        print("\n📊 Generating PPT-style decision plane (idealized)...")
        
        # 创建合成数据（模拟线性可分）
        np.random.seed(42)
        n_samples = 100
        
        # 生成两个线性可分的类别
        X1 = np.random.multivariate_normal(
            mean=[-1, -1, -1], 
            cov=np.eye(3)*0.3, 
            size=n_samples
        )
        X2 = np.random.multivariate_normal(
            mean=[1, 1, 1], 
            cov=np.eye(3)*0.3, 
            size=n_samples
        )
        
        X_synthetic = np.vstack([X1, X2])
        y_synthetic = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=45)
        
        # 绘制数据点（PPT风格）
        ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], 
                c='red', s=60, alpha=0.8, edgecolor='black',
                label='Class A', depthshade=True)
        ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], 
                c='blue', s=60, alpha=0.8, edgecolor='black',
                label='Class B', depthshade=True)
        
        # 绘制决策平面（线性）
        xx, yy = np.meshgrid(np.linspace(-2, 2, 10),
                            np.linspace(-2, 2, 10))
        
        # 创建平面：z = -x - y（简单的线性决策面）
        zz = -xx - yy
        
        ax.plot_surface(xx, yy, zz, 
                    alpha=0.4, color='gray',
                    linewidth=0.5, edgecolor='black')
        
        # 设置坐标轴
        ax.set_xlabel(f'{feature_names[0]}', fontsize=12, labelpad=10)
        ax.set_ylabel(f'{feature_names[1]}', fontsize=12, labelpad=10)
        ax.set_zlabel(f'{feature_names[2]}', fontsize=12, labelpad=10)
        
        # 标题
        ax.set_title('PPT Style: Idealized Decision Plane\n'
                    'Linear classifier separating two classes', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加说明文本
        ax.text2D(0.05, 0.95, 
                'Simplified illustration:\n• Red/Blue: Two classes\n• Gray plane: Decision boundary\n• Perfect linear separation',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}task2_ppt_style.png", 
                    dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()  

    def plot_ppt_style_probability_surface(self, feature_names):
        """
        绘制PPT风格的概率曲面示意图
        适用于任务3：二分类，三个特征
        """
        print("\n📊 Generating PPT-style probability surface (idealized)...")
        
        # 创建网格
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        xx, yy = np.meshgrid(x, y)
        
        # 创建S形概率曲面（理想的sigmoid形状）
        distance = np.sqrt(xx**2 + yy**2)
        zz = 1 / (1 + np.exp(-(distance - 2.5)))
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=45)
        
        # 绘制概率曲面（PPT风格）
        surf = ax.plot_surface(xx, yy, zz, 
                            cmap='RdYlBu',
                            alpha=0.85,
                            linewidth=0.1,
                            antialiased=True,
                            vmin=0, vmax=1)
        
        # 添加网格背景
        ax.plot_wireframe(xx, yy, np.zeros_like(xx), 
                        color='gray', alpha=0.3, linewidth=0.5)
        
        # 添加数据点示意
        np.random.seed(42)
        n_points = 30
        
        # 低概率区域点（蓝色）
        low_prob_points = np.random.uniform(-2.5, 0, (n_points, 3))
        low_prob_points[:, 2] = 0
        ax.scatter(low_prob_points[:, 0], low_prob_points[:, 1], low_prob_points[:, 2],
                c='blue', s=50, alpha=0.8, edgecolor='black', label='Class 0')
        
        # 高概率区域点（红色）
        high_prob_points = np.random.uniform(0, 2.5, (n_points, 3))
        high_prob_points[:, 2] = 0
        ax.scatter(high_prob_points[:, 0], high_prob_points[:, 1], high_prob_points[:, 2],
                c='red', s=50, alpha=0.8, edgecolor='black', label='Class 1')
        
        # 设置坐标轴
        ax.set_xlabel(f'{feature_names[0]}', fontsize=12, labelpad=10)
        ax.set_ylabel(f'{feature_names[1]}', fontsize=12, labelpad=10)
        ax.set_zlabel('Probability', fontsize=12, labelpad=10)
        
        # 颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Class 1 Probability', fontsize=11)
        
        # 决策边界线（概率=0.5）
        ax.contour(xx, yy, zz, 
                levels=[0.5], 
                colors='black', 
                linewidths=3,
                linestyles='--',
                alpha=0.8)
        
        # 标题
        ax.set_title('PPT Style: Idealized Probability Surface\n'
                    'Smooth transition from low to high probability', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}task3_ppt_style.png", 
                    dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()