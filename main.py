# main.py
import numpy as np
import matplotlib.pyplot as plt

# 导入配置
from config import ProjectConfig
from data_manager import IrisDataManager
from classifier_evaluator import ClassifierEvaluator

# 导入可视化模块
from visualization.base_visualizer import BaseVisualizer
from visualization.task1_2d_visualizer import Task1Visualizer
from visualization.task23_3d_visualizer import Task23Visualizer
from visualization.task4_3d_combined import Task4Visualizer
from visualization.performance_visualizer import PerformanceVisualizer

# 工具函数
from utils import create_output_dir

def main():
    """主程序"""
    print("="*70)
    print("IRIS DATA CLASSIFICATION AND VISUALIZATION PROJECT")
    print("="*70)
    
    try:
        # 1. 初始化
        print("\n🚀 Initializing system...")
        
        # 设置样式
        ProjectConfig.setup_style()
        
        # 创建输出目录
        create_output_dir(ProjectConfig.OUTPUT_DIR)
        
        # 初始化管理器
        data_manager = IrisDataManager()
        evaluator = ClassifierEvaluator()
        
        print(f"📊 Dataset: {data_manager.X.shape[0]} samples, {data_manager.X.shape[1]} features")
        print(f"   Classes: {', '.join(data_manager.target_names)}")
        
        # 2. 数据探索
        print("\n📈 Data exploration...")
        base_visualizer = BaseVisualizer(ProjectConfig)
        base_visualizer.plot_feature_distribution(data_manager)
        corr_matrix = base_visualizer.plot_correlation_analysis(data_manager)
        print("\n📊 Feature correlations:")
        print(corr_matrix.round(3))
        
        # 3. 任务1：2D可视化
        print("\n" + "="*60)
        print("TASK 1: 2D DECISION BOUNDARIES")
        print("="*60)
        
        X_2d, y_2d = data_manager.get_2d_data()
        feature_names_2d = data_manager.get_feature_names(ProjectConfig.FEATURES_2D)
        print(f"Features: {feature_names_2d[0]} vs {feature_names_2d[1]}")
        
        # 初始化任务1可视化器
        task1_visualizer = Task1Visualizer(ProjectConfig)
        
        # 2D决策边界
        performance_2d = task1_visualizer.plot_2d_decision_boundaries(
            X_2d, y_2d, evaluator.classifiers, feature_names_2d
        )
        
        # 评估分类器
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_2d, y_2d, test_size=ProjectConfig.TEST_SIZE,
            random_state=ProjectConfig.RANDOM_STATE, stratify=y_2d
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = evaluator.evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 性能对比
        performance_visualizer = PerformanceVisualizer(ProjectConfig)
        best_result = performance_visualizer.plot_performance_comparison(results)
        
        # 概率热图
        if best_result and best_result['classifier']:
            print(f"\n🎯 Detailed analysis with best classifier: {best_result['name']}")
            task1_visualizer.plot_probability_maps_fixed(
                X_2d, y_2d, best_result['classifier'], feature_names_2d
            )
        
        # 4. 任务2/3：3D可视化
        print("\n" + "="*60)
        print("TASK 2/3: 3D PROBABILITY SURFACE")
        print("="*60)
        
        task23_visualizer = Task23Visualizer(ProjectConfig)
        
        X_3d, y_3d = data_manager.get_3d_data(binary=True)
        feature_names_3d = data_manager.get_feature_names(ProjectConfig.FEATURES_3D)
        
        print(f"Features: {', '.join(feature_names_3d)}")
        print(f"Classes: setosa vs versicolor (binary)")
        
        # 3D概率曲面
        from sklearn.linear_model import LogisticRegression
        lr_clf = LogisticRegression(max_iter=1000, random_state=ProjectConfig.RANDOM_STATE)
        task23_visualizer.plot_3d_probability_surface(X_3d, y_3d, lr_clf, feature_names_3d)
        
        # 5. 任务4：3D边界+概率图
        print("\n" + "="*60)
        print("TASK 4: 3D DECISION BOUNDARIES + PROBABILITY HEATMAP")
        print("="*60)
        
        task4_visualizer = Task4Visualizer(ProjectConfig)
        
        X_3d_threeclass, y_3d_threeclass = data_manager.get_3d_data(binary=False)
        
        print(f"Features: {', '.join(feature_names_3d)}")
        print(f"Classes: {', '.join(data_manager.target_names)} (3 classes)")
        print(f"Classifier: Random Forest (supports probability prediction)")
        
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=ProjectConfig.RANDOM_STATE,
            max_depth=5
        )
        
        task4_fig = task4_visualizer.plot_3d_boundary_probability_combined(
            X_3d_threeclass, y_3d_threeclass, rf_clf, feature_names_3d
        )
        
        if task4_fig:
            print("✅ Task 4 completed successfully!")
            
        # 6. 总结
        print("\n" + "="*70)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\n📋 Summary:")
        print(f"   Classifiers tested: {len(evaluator.classifiers)}")
        print(f"   Output saved to: {ProjectConfig.OUTPUT_DIR}")
        
        if best_result:
            print(f"\n🎯 Best classifier: {best_result['name']}")
            print(f"   Accuracy: {best_result['accuracy']:.4f}")
        
        print(f"\n💡 Key findings:")
        print(f"   1. Petal features provide best class separation")
        print(f"   2. Ensemble methods show strong performance")
        print(f"   3. Decision boundaries are clearly visible in 2D/3D")
        
        print("\n📁 Generated files:")
        print(f"   • {ProjectConfig.OUTPUT_DIR}feature_distribution.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}correlation_analysis.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}2d_decision_boundaries.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}probability_maps_*.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}3d_probability_surface.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}task4_3d_boundary_probability.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}task4_2d_projections.png")
        print(f"   • {ProjectConfig.OUTPUT_DIR}performance_comparison.png")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()