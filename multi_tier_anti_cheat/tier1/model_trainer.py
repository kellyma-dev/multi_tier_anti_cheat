import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import os
import time
import logging
from typing import Dict, Tuple, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training and evaluation of multiple ML models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.evaluation_results = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get model configurations with hyperparameter grids.
        """
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
        """
        Optimize classification threshold to maximize recall while maintaining reasonable precision.
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Prioritize recall (weight it more heavily)
            # But ensure precision is not too low (at least 0.5)
            if precision >= 0.5:
                score = 0.7 * recall + 0.3 * f1  # Weighted score favoring recall
                threshold_results[threshold] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'score': score
                }
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return best_threshold, threshold_results.get(best_threshold, {})
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = configs[model_name]
        
        # Use RandomizedSearchCV for faster hyperparameter tuning
        # except for simpler models where GridSearchCV is feasible
        if model_name in ['logistic_regression']:
            search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='recall',  # Optimize for recall
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='recall',  # Optimize for recall
                n_jobs=-1,
                n_iter=20,  # Limit iterations for faster training
                random_state=42,
                verbose=1
            )
        
        # Fit the model
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Optimize threshold
        optimal_threshold, threshold_metrics = self.optimize_threshold(y_val, y_val_proba)
        y_val_pred_optimized = (y_val_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        val_metrics_optimized = self._calculate_metrics(y_val, y_val_pred_optimized)
        
        # Add AUC score
        val_metrics['auc'] = roc_auc_score(y_val, y_val_proba)
        val_metrics_optimized['auc'] = val_metrics['auc']  # Same AUC
        
        training_time = time.time() - start_time
        
        # Store model and results
        self.models[model_name] = {
            'model': best_model,
            'best_params': search.best_params_,
            'optimal_threshold': optimal_threshold,
            'training_time': training_time
        }
        
        results = {
            'model_name': model_name,
            'best_params': search.best_params_,
            'optimal_threshold': optimal_threshold,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_metrics_optimized': val_metrics_optimized,
            'threshold_metrics': threshold_metrics
        }
        
        # Extract feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance[model_name] = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            self.feature_importance[model_name] = np.abs(best_model.coef_[0])
        
        logger.info(f"Completed training {model_name} in {training_time:.2f}s")
        logger.info(f"Validation Recall (optimized): {val_metrics_optimized['recall']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train all models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with all training results
        """
        all_results = {}
        
        for model_name in self.get_model_configs().keys():
            try:
                results = self.train_model(model_name, X_train, y_train, X_val, y_val)
                all_results[model_name] = results
                self.evaluation_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select best model based on optimized recall
        best_recall = 0
        for model_name, results in all_results.items():
            recall = results['val_metrics_optimized']['recall']
            if recall > best_recall:
                best_recall = recall
                self.best_model_name = model_name
                self.best_model = self.models[model_name]['model']
        
        logger.info(f"Best model: {self.best_model_name} (Recall: {best_recall:.4f})")
        
        return all_results
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the best model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Test evaluation results
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        # Get optimal threshold
        optimal_threshold = self.models[self.best_model_name]['optimal_threshold']
        
        # Make predictions
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        test_metrics['auc'] = roc_auc_score(y_test, y_test_proba)
        
        # Detailed classification report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        results = {
            'model_name': self.best_model_name,
            'optimal_threshold': optimal_threshold,
            'test_metrics': test_metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"Test Results for {self.best_model_name}:")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {test_metrics['f1']:.4f}")
        logger.info(f"AUC: {test_metrics['auc']:.4f}")
        
        return results
    
    def save_models(self) -> None:
        """Save all trained models."""
        for model_name, model_data in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
            joblib.dump(model_data, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save best model separately for easy loading
        if self.best_model is not None:
            best_model_path = os.path.join(self.models_dir, "best_model.joblib")
            best_model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'optimal_threshold': self.models[self.best_model_name]['optimal_threshold'],
                'feature_importance': self.feature_importance.get(self.best_model_name)
            }
            joblib.dump(best_model_data, best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
    
    def load_best_model(self) -> Dict:
        """Load the best trained model."""
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        if os.path.exists(best_model_path):
            return joblib.load(best_model_path)
        else:
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    def get_feature_importance_analysis(self, feature_names: List[str]) -> Dict:
        """
        Analyze feature importance across models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Feature importance analysis
        """
        if not self.feature_importance:
            return {}
        
        importance_df = pd.DataFrame(self.feature_importance, index=feature_names)
        
        # Calculate average importance across models
        importance_df['average'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('average', ascending=False)
        
        analysis = {
            'feature_importance_by_model': importance_df.to_dict(),
            'most_important_features': importance_df.head(3).index.tolist(),
            'least_important_features': importance_df.tail(3).index.tolist()
        }
        
        return analysis


def main():
    """Test the model trainer."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    train_df = loader.load_training_data()
    val_df = loader.load_validation_data()
    
    # Preprocess data
    X_train, y_train = loader.preprocess_data(train_df, fit_scaler=True)
    X_val, y_val = loader.preprocess_data(val_df, fit_scaler=False)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Print results summary
    print("\nModel Comparison (Optimized Thresholds):")
    print("-" * 80)
    for model_name, result in results.items():
        metrics = result['val_metrics_optimized']
        print(f"{model_name:20} | Recall: {metrics['recall']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f}")
    
    # Save models
    trainer.save_models()
    
    # Feature importance analysis
    importance_analysis = trainer.get_feature_importance_analysis(loader.get_feature_names())
    print(f"\nFeature Importance Analysis:")
    print(f"Most important: {importance_analysis.get('most_important_features', [])}")
    print(f"Least important: {importance_analysis.get('least_important_features', [])}")


if __name__ == "__main__":
    main()
