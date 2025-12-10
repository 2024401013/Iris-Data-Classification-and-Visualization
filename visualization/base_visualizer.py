# visualization/base_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

class BaseVisualizer:
    """基础可视化类（特征分布、相关性）"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_feature_distribution(self, data_manager):
        """特征分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Iris Feature Distribution Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        colors = [self.config.COLORS[name] for name in data_manager.target_names]
        
        for idx, feature in enumerate(data_manager.feature_names):
            ax = axes[idx // 2, idx % 2]
            
            # Violin plot (original)
            parts = ax.violinplot([data_manager.df[data_manager.df['species'] == species][feature] 
                                  for species in data_manager.target_names],
                                 showmeans=True, showmedians=True)
            
            # Set colors (original)
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
            
            # Add scatter points
            for i, species in enumerate(data_manager.target_names):
                species_data = data_manager.df[data_manager.df['species'] == species][feature]
                x_pos = np.random.normal(i + 1, 0.04, size=len(species_data))
                ax.scatter(x_pos, species_data, alpha=0.5, s=20, 
                          color='black', edgecolor='none')
            
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(data_manager.target_names, rotation=45)
            ax.set_title(feature, fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('cm', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}feature_distribution.png", 
                       dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self, data_manager):
        """相关性分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Correlation heatmap
        corr_matrix = data_manager.df.iloc[:, :4].corr()
        im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax1.set_xticks(range(4))
        ax1.set_yticks(range(4))
        ax1.set_xticklabels(['SL', 'SW', 'PL', 'PW'], fontsize=11)
        ax1.set_yticklabels(['Sepal Length', 'Sepal Width', 
                           'Petal Length', 'Petal Width'], fontsize=11)
        
        ax1.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        for i in range(4):
            for j in range(4):
                value = corr_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # Scatter plot with best features
        colors = [self.config.COLORS[name] for name in data_manager.target_names]
        feat1, feat2 = self.config.FEATURES_2D
        
        for i, species in enumerate(data_manager.target_names):
            mask = data_manager.df['species'] == species
            ax2.scatter(data_manager.df.loc[mask, data_manager.feature_names[feat1]], 
                       data_manager.df.loc[mask, data_manager.feature_names[feat2]],
                       c=colors[i], label=species, s=50, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel(data_manager.feature_names[feat1], fontsize=12)
        ax2.set_ylabel(data_manager.feature_names[feat2], fontsize=12)
        ax2.set_title('Feature Scatter Plot (Selected for 2D Visualization)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}correlation_analysis.png", 
                       dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    