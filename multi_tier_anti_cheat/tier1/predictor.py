"""
Prediction module for anti-cheat ML detection system.
Provides easy interface for making predictions on new data.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Union, List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AntiCheatPredictor:
    """Predictor class for anti-cheat detection."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.model_data = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self, model_path: str = None) -> None:
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to model file (defaults to best_model.joblib)
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, "best_model.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data
        self.model_data = joblib.load(model_path)
        logger.info(f"Loaded model: {self.model_data['model_name']}")
        logger.info(f"Optimal threshold: {self.model_data['optimal_threshold']:.4f}")
        
        # Load scaler (should be saved with data_loader)
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded data scaler")
        else:
            logger.warning("Scaler not found. You may need to provide pre-scaled data.")
    
    def predict_single(self, features: Union[List, np.ndarray]) -> Dict:
        """
        Make prediction for a single sample.
        
        Args:
            features: Feature values (should match training feature order)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model_data is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get probability scores
        probabilities = self.model_data['model'].predict_proba(X)[0]
        cheater_probability = probabilities[1]
        
        # Make prediction using optimal threshold
        is_cheater = cheater_probability >= self.model_data['optimal_threshold']
        
        return {
            'is_cheater': bool(is_cheater),
            'cheater_probability': float(cheater_probability),
            'confidence': float(max(probabilities)),
            'threshold_used': float(self.model_data['optimal_threshold'])
        }
    
    def predict_batch(self, features: Union[List[List], np.ndarray, pd.DataFrame]) -> List[Dict]:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Feature matrix (samples x features)
            
        Returns:
            List of prediction dictionaries
        """
        if self.model_data is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = np.array(features)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get probability scores
        probabilities = self.model_data['model'].predict_proba(X)
        cheater_probabilities = probabilities[:, 1]
        
        # Make predictions using optimal threshold
        predictions = cheater_probabilities >= self.model_data['optimal_threshold']
        
        # Format results
        results = []
        for i in range(len(X)):
            results.append({
                'is_cheater': bool(predictions[i]),
                'cheater_probability': float(cheater_probabilities[i]),
                'confidence': float(max(probabilities[i])),
                'threshold_used': float(self.model_data['optimal_threshold'])
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model_data is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return {
            'model_name': self.model_data['model_name'],
            'optimal_threshold': self.model_data['optimal_threshold'],
            'feature_importance': self.model_data.get('feature_importance'),
            'model_type': type(self.model_data['model']).__name__
        }
    
    def explain_prediction(self, features: Union[List, np.ndarray]) -> Dict:
        """
        Provide explanation for a prediction (simple feature contribution).
        
        Args:
            features: Feature values
            
        Returns:
            Dictionary with prediction explanation
        """
        prediction = self.predict_single(features)
        
        # Get feature importance if available
        feature_importance = self.model_data.get('feature_importance')
        if feature_importance is not None:
            # Calculate feature contributions (simplified)
            X = np.array(features)
            if self.scaler is not None:
                X = self.scaler.transform(X.reshape(1, -1))[0]
            
            contributions = X * feature_importance
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(contributions))[-3:][::-1]
            
            explanation = {
                'prediction': prediction,
                'top_contributing_features': [
                    {
                        'feature_index': int(idx),
                        'contribution': float(contributions[idx]),
                        'feature_value': float(X[idx])
                    }
                    for idx in top_indices
                ]
            }
        else:
            explanation = {
                'prediction': prediction,
                'note': 'Feature importance not available for this model type'
            }
        
        return explanation


def main():
    """Test the predictor."""
    predictor = AntiCheatPredictor()
    
    try:
        # Load model
        predictor.load_model()
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"Loaded model: {model_info}")
        
        # Example prediction (using dummy data)
        # In real use, these would be actual game metrics
        example_features = [5.7e-08, 2.3e-08, 1.2e-15, 3.4e-14, 1.5e-16, 7.6e-08, 2.8e-31]
        
        result = predictor.predict_single(example_features)
        print(f"\nExample prediction: {result}")
        
        # Explanation
        explanation = predictor.explain_prediction(example_features)
        print(f"\nPrediction explanation: {explanation}")
        
    except FileNotFoundError:
        print("No trained model found. Please run train_evaluate_models.py first to train models.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
