# visualization/performance_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

class PerformanceVisualizer:
    """性能对比可视化"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_performance_comparison(self, results):
        """Performance comparison with professional styling"""
        if not results:
            print("❌ No results to display")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Classifier Performance Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Extract data
        names = [r['name'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        cv_means = [r['cv_mean'] for r in results]
        cv_stds = [r['cv_std'] for r in results]
        
        # Professional color gradient
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        
        # Plot 1: Accuracy ranking
        y_pos = np.arange(len(names))
        bars1 = axes[0, 0].barh(y_pos, accuracies, color=colors, height=0.7)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(names, fontsize=10)
        axes[0, 0].set_xlabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Classifier Accuracy Ranking', 
                           fontsize=13, fontweight='bold', pad=15)
        axes[0, 0].set_xlim([0, 1.05])
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Add values
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{acc:.4f}', ha='left', va='center', 
                          fontweight='bold', fontsize=9)
        
        # Plot 2: CV vs Test accuracy
        x_pos = np.arange(len(names))
        axes[0, 1].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                      color=colors, alpha=0.7)
        axes[0, 1].plot(x_pos, accuracies, 'r-o', linewidth=2, 
                       markersize=8, label='Test Accuracy')
        
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([n[:15] for n in names], rotation=45, 
                                  ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Cross-validation vs Test Accuracy', 
                           fontsize=13, fontweight='bold', pad=15)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Composite score
        stability_scores = [1 - std for std in cv_stds]
        composite_scores = [acc * stab for acc, stab in zip(accuracies, stability_scores)]
        composite_scores = np.array(composite_scores) / max(composite_scores)
        
        bars3 = axes[1, 0].bar(x_pos, composite_scores, color=colors, alpha=0.7)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([n[:15] for n in names], rotation=45, 
                                  ha='right', fontsize=9)
        axes[1, 0].set_ylabel('Composite Score', fontsize=12)
        axes[1, 0].set_title('Classifier Composite Performance Score', 
                           fontsize=13, fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Mark best classifier
        best_idx = np.argmax(composite_scores)
        axes[1, 0].annotate('🏆 Best', 
                           xy=(best_idx, composite_scores[best_idx]),
                           xytext=(best_idx, composite_scores[best_idx] + 0.05),
                           arrowprops=dict(facecolor='red', shrink=0.05),
                           fontsize=11, fontweight='bold', ha='center')
        
        # Plot 4: Accuracy vs Stability
        scatter = axes[1, 1].scatter(accuracies, stability_scores, s=100, 
                                    c=colors, alpha=0.7, edgecolors='black')
        
        # Add labels with better positioning
        for i, (name, acc, stab) in enumerate(zip(names, accuracies, stability_scores)):
            offset_x = 0.005 if i % 2 == 0 else -0.005
            offset_y = 0.005 if i < len(names)/2 else -0.005
            axes[1, 1].annotate(name[:10], (acc + offset_x, stab + offset_y),
                               fontsize=8, ha='center')
        
        axes[1, 1].set_xlabel('Accuracy', fontsize=12)
        axes[1, 1].set_ylabel('Stability (1 - CV std)', fontsize=12)
        axes[1, 1].set_title('Accuracy vs Stability', 
                           fontsize=13, fontweight='bold', pad=15)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config.SAVE_FIGURES:
            plt.savefig(f"{self.config.OUTPUT_DIR}performance_comparison.png", 
                       dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        
        # Print best classifier
        best_result = results[best_idx]
        print("\n" + "="*60)
        print("🏆 BEST CLASSIFIER ANALYSIS")
        print("="*60)
        print(f"Classifier: {best_result['name']}")
        print(f"Test Accuracy: {best_result['accuracy']:.4f}")
        print(f"Cross-validation: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        print("="*60)
        
        return best_result
