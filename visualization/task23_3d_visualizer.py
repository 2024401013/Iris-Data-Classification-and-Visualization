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
    