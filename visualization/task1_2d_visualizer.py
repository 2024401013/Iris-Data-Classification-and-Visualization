# visualization/task1_2d_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class Task1Visualizer:
    """任务1：2D决策边界和概率图"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_2d_decision_boundaries(self, X, y, classifier_dict, feature_names):
        """2D决策边界对比"""
        # Data preparation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create grid
        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Create figure
        n_classifiers = len(classifier_dict)
        n_cols = 3
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        fig.suptitle(f'2D Decision Boundary Comparison\nFeatures: {feature_names[0]} vs {feature_names[1]}',
                    fontsize=16, fontweight='bold', y=1.02)
        
        axes = axes.flatten()
        
        # Enhanced colors - more saturated
        class_colors = [self.config.COLORS['setosa'], 
                       self.config.COLORS['versicolor'], 
                       self.config.COLORS['virginica']]
        
        # Brighter background colors
        background_colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF']  # Bright red, green, blue
        
        performance = []
        
        for idx, (name, clf) in enumerate(classifier_dict.items()):
            ax = axes[idx]
            
            try:
                # Train
                clf.fit(X_train_scaled, y_train)
                accuracy = clf.score(X_test_scaled, y_test)
                performance.append((name, accuracy))
                
                # Predict
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision regions with BRIGHTER colors
                cmap_light = ListedColormap(background_colors)
                ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)  # Increased alpha
                
                # Plot data points with ENHANCED visibility
                for i in range(3):
                    mask = y_train == i
                    if np.sum(mask) > 0:
                        # Larger markers, darker edges
                        ax.scatter(X_train_scaled[mask, 0], X_train_scaled[mask, 1],
                                  c=class_colors[i], s=40, alpha=0.9,  # Increased size and alpha
                                  edgecolor='black', linewidth=1.2)
                
                ax.set_xlabel(f'{feature_names[0]}', fontsize=10)
                ax.set_ylabel(f'{feature_names[1]}', fontsize=10)
                ax.set_title(f'{name}\nAccuracy: {accuracy:.3f}', 
                           fontsize=11, fontweight='bold', pad=12)
                ax.grid(True, alpha=0.3)
                
                # Add legend to first plot only
                if idx == 0:
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=class_colors[0], edgecolor='black', label='setosa'),
                        Patch(facecolor=class_colors[1], edgecolor='black', label='versicolor'),
                        Patch(facecolor=class_colors[2], edgecolor='black', label='virginica')
                    ]
                    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
                
            except Exception as e:
                print(f"  Training {name} failed: {str(e)[:50]}")
                ax.text(0.5, 0.5, 'Training Failed', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{name}\nFailed', fontsize=11)
                ax.grid(True, alpha=0.3)
                performance.append((name, 0.0))
        
        # Hide empty subplots
        for idx in range(len(classifier_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}2d_decision_boundaries.png", 
                       dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        
        return performance
    
    
    def plot_probability_maps_fixed(self, X, y, classifier, feature_names):
        """概率热图"""
        print(f"\nGenerating probability maps for {classifier.__class__.__name__}...")
        
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
        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        
        # Use coarser grid for faster computation
        grid_resolution = 80
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                            np.linspace(y_min, y_max, grid_resolution))
        
        # Check if classifier supports probability
        if not hasattr(classifier, 'predict_proba'):
            print(f"⚠️ {classifier.__class__.__name__} does not support probability prediction")
            return
        
        try:
            # Get probabilities
            probs = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Probability Maps - {classifier.__class__.__name__}\nAccuracy: {accuracy:.3f}', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            # Plot 1: Overall decision boundaries
            ax1 = axes[0, 0]
            
            # Get predictions for overall plot
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Use imshow for cleaner decision regions (from your reference)
            ax1.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), 
                      origin='lower',
                      cmap=ListedColormap([self.config.COLORS['setosa'],
                                          self.config.COLORS['versicolor'],
                                          self.config.COLORS['virginica']]),
                      alpha=0.7, aspect='auto')
            
            # Plot data points
            colors = [self.config.COLORS['setosa'], 
                     self.config.COLORS['versicolor'], 
                     self.config.COLORS['virginica']]
            
            for i in range(3):
                mask = y_train == i
                if np.sum(mask) > 0:
                    ax1.scatter(X_train_scaled[mask, 0], X_train_scaled[mask, 1],
                              c=colors[i], edgecolors='k', s=40, alpha=0.8)
            
            ax1.set_xlabel(f'{feature_names[0]} (standardized)', fontsize=11)
            ax1.set_ylabel(f'{feature_names[1]} (standardized)', fontsize=11)
            ax1.set_title('Overall Decision Boundaries', fontsize=13, fontweight='bold', pad=12)
            ax1.grid(True, alpha=0.3)
            
            # Plots 2-4: Individual class probabilities
            class_names = ['setosa', 'versicolor', 'virginica']
            plot_positions = [(0, 1), (1, 0), (1, 1)]
            
            for i, (row, col) in enumerate(plot_positions):
                ax = axes[row, col]
                
                if i < probs.shape[1]:  # Ensure we have enough classes
                    # Get probabilities for this class
                    prob_map = probs[:, i].reshape(xx.shape)
                    
                    # Create custom colormap
                    cmap = LinearSegmentedColormap.from_list(
                        f'class_{i}', 
                        ['white', colors[i]], 
                        N=256
                    )
                    
                    # FIXED: Use np.linspace to create explicit levels
                    levels = np.linspace(0, 1, 21)  # 20 levels from 0 to 1
                    
                    # Plot with explicit levels to avoid len() error
                    contour = ax.contourf(xx, yy, prob_map, 
                                         levels=levels,  # Explicit levels
                                         cmap=cmap,
                                         alpha=0.8, 
                                         extend='both')
                    
                    # Plot data points
                    for j in range(3):
                        mask = y_train == j
                        if np.sum(mask) > 0:
                            ax.scatter(X_train_scaled[mask, 0], X_train_scaled[mask, 1],
                                      c=colors[j], edgecolors='k', s=30, alpha=0.8)
                    
                    # Add colorbar
                    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
                    cbar.set_label('Probability', fontsize=10)
                    
                    ax.set_xlabel(f'{feature_names[0]} (standardized)', fontsize=11)
                    ax.set_ylabel(f'{feature_names[1]} (standardized)', fontsize=11)
                    ax.set_title(f'{class_names[i].title()} Probability', 
                               fontsize=13, fontweight='bold', pad=12)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            if self.config.SAVE_FIGURES:
                plt.savefig(f"{self.config.OUTPUT_DIR}probability_maps_{classifier.__class__.__name__}.png", 
                           dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generating probability maps: {e}")
            # Create simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'Probability Map Generation Failed\nError: {str(e)[:50]}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Probability Maps - {classifier.__class__.__name__}', fontsize=14)
            plt.show()
    