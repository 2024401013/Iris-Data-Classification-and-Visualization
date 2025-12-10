# classifier_evaluator.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# 导入各种分类器
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

class ClassifierEvaluator:
    """Multi-classifier training and evaluation"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifiers = self._init_classifiers()
        
    def _init_classifiers(self):
        """初始化分类器集合"""
        from config import ProjectConfig

        return {
            'Logistic Regression': LogisticRegression(
                multi_class='multinomial', max_iter=1000, 
                random_state=ProjectConfig.RANDOM_STATE),
            'SVM (Linear)': SVC(kernel='linear', probability=True, 
                               random_state=ProjectConfig.RANDOM_STATE),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, 
                            random_state=ProjectConfig.RANDOM_STATE),
            'Decision Tree': DecisionTreeClassifier(max_depth=4, 
                                                   random_state=ProjectConfig.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(n_estimators=100, 
                                                   random_state=ProjectConfig.RANDOM_STATE),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                           random_state=ProjectConfig.RANDOM_STATE),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, 
                                          random_state=ProjectConfig.RANDOM_STATE),
            'Naive Bayes': GaussianNB(),
            'K-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
        }
    
    def evaluate_all(self, X_train, X_test, y_train, y_test):
        """评估所有分类器"""
        from config import ProjectConfig
        
        results = []
        
        for name, clf in self.classifiers.items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                cv_scores = cross_val_score(
                    clf, np.vstack([X_train, X_test]), 
                    np.hstack([y_train, y_test]), 
                    cv=ProjectConfig.CV_FOLDS
                )
                
                results.append({
                    'name': name,
                    'classifier': clf,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                })
                
            except Exception as e:
                print(f"❌ {name:20s} | Error: {str(e)[:50]}")
                results.append({
                    'name': name, 'classifier': None,
                    'accuracy': 0, 'cv_mean': 0, 'cv_std': 0
                })
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        return results