# visualization/task4_3d_combined.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

class Task4Visualizer:
    """任务4：3D边界+概率图"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_3d_boundary_probability_combined(self, X, y, classifier, feature_names):
        """
        3D边界+概率图
        """
        print(f"\n📊 Task 4: Generating 3D Boundary + Probability Map (3 classes, 3 features)...")
        
        # 数据准备
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练
        classifier.fit(X_train_scaled, y_train)
        accuracy = classifier.score(X_test_scaled, y_test)
        print(f"  Classifier: {classifier.__class__.__name__}, Accuracy: {accuracy:.3f}")
        
        # 检查是否支持概率预测
        if not hasattr(classifier, 'predict_proba'):
            print(f"⚠️ {classifier.__class__.__name__} does not support probability prediction")
            return self._plot_without_probabilities(X_train_scaled, y_train, classifier, 
                                                feature_names, accuracy)
        
        try:
            # 测试概率预测
            test_probs = classifier.predict_proba(X_test_scaled[:2])
            if test_probs is None or test_probs.shape[1] != 3:
                raise ValueError("predict_proba returned invalid shape")
        except Exception as e:
            print(f"❌ Probability prediction failed: {e}")
            return self._plot_without_probabilities(X_train_scaled, y_train, classifier,
                                                feature_names, accuracy)
        
        # 创建3D网格
        grid_resolution = 40  # 适中的分辨率
        x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
        y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
        z_min, z_max = X_train_scaled[:, 2].min() - 0.5, X_train_scaled[:, 2].max() + 0.5
        
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
            np.linspace(z_min, z_max, grid_resolution)
        )
        
        # 获取网格点的概率预测
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        probs = classifier.predict_proba(grid_points)
        
        # 计算最大概率和对应的类别
        max_probs = np.max(probs, axis=1)
        pred_classes = np.argmax(probs, axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 9))
        
        # ====== 子图1：决策边界（使用替代方法） ======
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=25, azim=45)
        
        # 类别颜色
        class_colors = [self.config.COLORS['setosa'], 
                    self.config.COLORS['versicolor'], 
                    self.config.COLORS['virginica']]
        
        # 边界名称
        boundary_names = [
            'setosa vs versicolor',
            'setosa vs virginica', 
            'versicolor vs virginica'
        ]
        
        # 类别对
        class_pairs = [(0, 1), (0, 2), (1, 2)]
        boundary_colors = ['#FF6B6B80', '#4ECDC480', '#45B7D180']
        
        # 使用替代方法绘制决策边界（不依赖marching_cubes）
        print("  Using alternative boundary visualization...")
        
        # 方法1：绘制概率接近的边界点
        boundaries_drawn = 0
        for idx, (i, j) in enumerate(class_pairs):
            # 计算概率差
            prob_diff = np.abs(probs[:, i] - probs[:, j])
            
            # 找到概率接近的点（边界区域）
            boundary_mask = prob_diff < 0.1  # 概率差小于0.1
            
            if np.sum(boundary_mask) > 0:
                # 采样显示边界点
                n_boundary_points = min(2000, np.sum(boundary_mask))
                boundary_indices = np.where(boundary_mask)[0]
                if len(boundary_indices) > n_boundary_points:
                    boundary_indices = np.random.choice(boundary_indices, n_boundary_points, replace=False)
                
                # 绘制边界点（使用半透明）
                ax1.scatter(
                    xx.ravel()[boundary_indices],
                    yy.ravel()[boundary_indices],
                    zz.ravel()[boundary_indices],
                    color=boundary_colors[idx],
                    s=15,  # 固定大小
                    alpha=0.3,  # 固定透明度
                    edgecolors='none',
                    label=boundary_names[idx]
                )
                
                boundaries_drawn += 1
                print(f"  ✓ Showing {len(boundary_indices)} boundary points for {boundary_names[idx]}")
        
        # 绘制训练数据点
        for i in range(3):
            mask = y_train == i
            if np.sum(mask) > 0:
                ax1.scatter(X_train_scaled[mask, 0], X_train_scaled[mask, 1], X_train_scaled[mask, 2],
                        c=class_colors[i], s=80, alpha=0.9, edgecolor='black', linewidth=1.5,
                        label=['setosa', 'versicolor', 'virginica'][i],
                        zorder=10)  # 确保数据点在顶部
        
        ax1.set_xlabel(f'{feature_names[0]} (standardized)', fontsize=12, labelpad=12)
        ax1.set_ylabel(f'{feature_names[1]} (standardized)', fontsize=12, labelpad=12)
        ax1.set_zlabel(f'{feature_names[2]} (standardized)', fontsize=12, labelpad=12)
        ax1.set_title(f'3D Decision Boundaries (Boundary Points)\n{classifier.__class__.__name__}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 改进的图例（只显示实际绘制的项目）
        from matplotlib.patches import Patch
        legend_elements = []
        
        # 添加数据点图例
        for i in range(3):
            legend_elements.append(
                Patch(facecolor=class_colors[i], edgecolor='black', 
                    label=['setosa', 'versicolor', 'virginica'][i])
            )
        
        # 添加边界图例
        for idx in range(min(boundaries_drawn, 3)):
            legend_elements.append(
                Patch(facecolor=boundary_colors[idx], alpha=0.3,
                    label=boundary_names[idx])
            )
        
        if legend_elements:
            ax1.legend(handles=legend_elements, fontsize=9, loc='upper right', ncol=2)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_zlim([z_min, z_max])
        
        # ====== 子图2：改进的概率热力图（修复alpha问题） ======
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=25, azim=45)
        
        # 采样显示（避免内存问题）
        n_points = min(10000, len(grid_points))
        sample_indices = np.random.choice(len(grid_points), n_points, replace=False)
        
        # 为每个采样点创建颜色（根据概率）
        colors_rgba = []
        
        for idx in sample_indices:
            class_idx = pred_classes[idx]
            prob_val = max_probs[idx]
            
            # 获取基础颜色
            if class_idx == 0:  # setosa
                base_color = np.array([1.0, 0.42, 0.42])  # 珊瑚红
            elif class_idx == 1:  # versicolor
                base_color = np.array([0.31, 0.80, 0.77])  # 青绿色
            else:  # virginica
                base_color = np.array([0.27, 0.72, 0.82])  # 天空蓝
            
            # 根据概率调整颜色饱和度
            # 低概率（0.3-0.5）: 颜色较淡，高概率（0.9-1.0）: 颜色饱和
            saturation = 0.3 + 0.7 * prob_val
            
            # 根据概率调整透明度
            # 高置信度区域更不透明，边界区域更透明
            alpha = 0.2 + 0.6 * prob_val
            
            # 应用饱和度
            color_rgb = base_color * saturation
            
            # 限制在[0,1]范围内
            color_rgb = np.clip(color_rgb, 0, 1)
            
            # 创建RGBA颜色
            color_rgba = [color_rgb[0], color_rgb[1], color_rgb[2], alpha]
            colors_rgba.append(color_rgba)
        
        colors_rgba = np.array(colors_rgba)
        
        # 绘制概率热力图 - 使用固定alpha，通过颜色本身控制透明度
        # 方法1：分组绘制不同置信度的点
        confidence_levels = [(0.3, 0.5, 0.1, 5),   # 低置信度：小点，低透明度
                            (0.5, 0.7, 0.2, 10),  # 中置信度
                            (0.7, 0.9, 0.4, 15),  # 高置信度
                            (0.9, 1.0, 0.7, 20)]  # 很高置信度
        
        for min_conf, max_conf, alpha, size in confidence_levels:
            # 选择该置信度范围内的点
            conf_mask = (max_probs[sample_indices] >= min_conf) & (max_probs[sample_indices] < max_conf)
            if np.sum(conf_mask) > 0:
                conf_indices = sample_indices[conf_mask]
                conf_colors = colors_rgba[conf_mask, :3]  # 只取RGB
                
                ax2.scatter(
                    xx.ravel()[conf_indices],
                    yy.ravel()[conf_indices],
                    zz.ravel()[conf_indices],
                    c=conf_colors,
                    s=size,
                    alpha=alpha,  # 固定alpha
                    edgecolors='none'
                )
        
        # 绘制训练数据点（突出显示）
        for i in range(3):
            mask = y_train == i
            if np.sum(mask) > 0:
                ax2.scatter(X_train_scaled[mask, 0], X_train_scaled[mask, 1], X_train_scaled[mask, 2],
                        c=class_colors[i], s=100, alpha=1.0, 
                        edgecolor='black', linewidth=2.0, zorder=100,
                        label=['setosa', 'versicolor', 'virginica'][i])
        
        ax2.set_xlabel(f'{feature_names[0]} (standardized)', fontsize=12, labelpad=12)
        ax2.set_ylabel(f'{feature_names[1]} (standardized)', fontsize=12, labelpad=12)
        ax2.set_zlabel(f'{feature_names[2]} (standardized)', fontsize=12, labelpad=12)
        ax2.set_title(f'3D Probability Heatmap\nColor = Class, Opacity = Confidence', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements_2 = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.config.COLORS['setosa'], 
                markersize=10, label='Setosa region', alpha=1.0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.config.COLORS['versicolor'], 
                markersize=10, label='Versicolor region', alpha=1.0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.config.COLORS['virginica'], 
                markersize=10, label='Virginica region', alpha=1.0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markersize=5, label='Low confidence', alpha=0.2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markersize=10, label='Medium confidence', alpha=0.4),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markersize=15, label='High confidence', alpha=0.7),
        ]
        
        ax2.legend(handles=legend_elements_2, fontsize=9, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        ax2.set_zlim([z_min, z_max])
        
        # 主标题
        plt.suptitle(
            f'Task 4: 3D Decision Boundaries + Probability Heatmap\n'
            f'Features: {feature_names[0]}, {feature_names[1]}, {feature_names[2]} | '
            f'Classifier: {classifier.__class__.__name__} (Accuracy: {accuracy:.3f})',
            fontsize=16, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        
        if self.config.SAVE_FIGURES:
            filename = f"{self.config.OUTPUT_DIR}task4_3d_boundary_probability.png"
            plt.savefig(filename, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"  ✅ Saved to: {filename}")
        
        plt.show()
        
        # 创建单独的2D概率投影图（作为补充）
        self._plot_2d_probability_projections(xx, yy, zz, probs, max_probs, pred_classes,
                                            X_train_scaled, y_train, feature_names,
                                            classifier.__class__.__name__, accuracy)
        
        # 打印概率统计
        print(f"\n📊 Probability statistics:")
        print(f"  Max probability range: {max_probs.min():.3f} - {max_probs.max():.3f}")
        print(f"  Average confidence: {max_probs.mean():.3f}")
        
        confidence_levels_stats = [0.5, 0.7, 0.9]
        for level in confidence_levels_stats:
            confident_mask = max_probs >= level
            confident_ratio = np.sum(confident_mask) / len(max_probs)
            print(f"  Points with confidence ≥{level}: {confident_ratio:.1%}")
        
        for i in range(3):
            class_mask = pred_classes == i
            if np.sum(class_mask) > 0:
                class_name = ['setosa', 'versicolor', 'virginica'][i]
                print(f"  Class {i} ({class_name}):")
                print(f"    Predicted proportion: {np.sum(class_mask)/len(pred_classes):.3f}")
                print(f"    Average confidence: {max_probs[class_mask].mean():.3f}")
                print(f"    High confidence (≥0.9): {np.sum(max_probs[class_mask] >= 0.9)/np.sum(class_mask):.1%}")
        
        return fig

    
    def _plot_2d_probability_projections(self, xx, yy, zz, probs, max_probs, pred_classes,
                                    X_train, y_train, feature_names, clf_name, accuracy):
        """绘制2D投影作为补充"""
        print("  Generating 2D probability projections as supplementary views...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'2D Probability Projections - {clf_name}\nAccuracy: {accuracy:.3f}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 采样点
        n_samples = min(5000, len(xx.ravel()))
        sample_idx = np.random.choice(len(xx.ravel()), n_samples, replace=False)
        
        # 2D投影平面
        projections = [
            (0, 1, 'XY Plane'),
            (0, 2, 'XZ Plane'), 
            (1, 2, 'YZ Plane')
        ]
        
        class_colors = [self.config.COLORS['setosa'], 
                    self.config.COLORS['versicolor'], 
                    self.config.COLORS['virginica']]
        
        for row in range(2):
            for col in range(3):
                ax = axes[row, col]
                proj_idx = row * 3 + col
                
                if proj_idx < len(projections):
                    i, j, title = projections[proj_idx]
                    
                    # 第一行：类别区域
                    if row == 0:
                        # 绘制预测类别
                        for class_idx in range(3):
                            mask = pred_classes[sample_idx] == class_idx
                            if np.sum(mask) > 0:
                                ax.scatter(
                                    [xx.ravel()[sample_idx][mask], yy.ravel()[sample_idx][mask], zz.ravel()[sample_idx][mask]][i],
                                    [xx.ravel()[sample_idx][mask], yy.ravel()[sample_idx][mask], zz.ravel()[sample_idx][mask]][j],
                                    c=class_colors[class_idx], s=10, alpha=0.6, 
                                    label=['setosa', 'versicolor', 'virginica'][class_idx]
                                )
                        
                        # 绘制训练数据
                        for class_idx in range(3):
                            mask = y_train == class_idx
                            if np.sum(mask) > 0:
                                ax.scatter(
                                    X_train[mask, i], X_train[mask, j],
                                    c=class_colors[class_idx], s=60, alpha=1.0,
                                    edgecolor='black', linewidth=1.5, zorder=10
                                )
                        
                        ax.set_title(f'{title}\nPredicted Class Regions', fontsize=12, fontweight='bold')
                        if col == 0:
                            ax.legend(fontsize=9, loc='upper right')
                    
                    # 第二行：置信度热图
                    else:
                        # 根据最大概率着色
                        scatter = ax.scatter(
                            [xx.ravel()[sample_idx], yy.ravel()[sample_idx], zz.ravel()[sample_idx]][i],
                            [xx.ravel()[sample_idx], yy.ravel()[sample_idx], zz.ravel()[sample_idx]][j],
                            c=max_probs[sample_idx], cmap='viridis', s=15, alpha=0.7,
                            vmin=0.3, vmax=1.0
                        )
                        
                        # 绘制训练数据
                        for class_idx in range(3):
                            mask = y_train == class_idx
                            if np.sum(mask) > 0:
                                ax.scatter(
                                    X_train[mask, i], X_train[mask, j],
                                    c=class_colors[class_idx], s=50, alpha=1.0,
                                    edgecolor='white', linewidth=1.0
                                )
                        
                        ax.set_title(f'{title}\nPrediction Confidence', fontsize=12, fontweight='bold')
                        
                        if col == 2:
                            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                            cbar.set_label('Max Probability', fontsize=10)
                    
                    ax.set_xlabel(feature_names[i], fontsize=10)
                    ax.set_ylabel(feature_names[j], fontsize=10)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        
        if self.config.SAVE_FIGURES:
            filename = f"{self.config.OUTPUT_DIR}task4_2d_projections.png"
            plt.savefig(filename, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"  ✅ 2D projections saved to: {filename}")
        
        plt.show()
