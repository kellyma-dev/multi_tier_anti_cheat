# Anti-Cheat ML Detection System - Model Documentation

## Overview

This system implements a machine learning pipeline for detecting cheating behavior in gaming environments. It uses ensemble methods and multiple algorithms to classify players as cheaters or legitimate players based on gameplay metrics.

## Quick Start

### Installation

```bash
# Install required dependencies
poetry setup
```

### Basic Usage

```python
# Train models
poetry run python model_trainer.py

# Make predictions on new data
from predictor import AntiCheatPredictor

predictor = AntiCheatPredictor()
predictor.load_model()

# Single prediction
features = [5.7e-08, 2.3e-08, 1.2e-15, 3.4e-14, 1.5e-16, 7.6e-08, 2.8e-31]
result = predictor.predict_single(features)
print(f"Is cheater: {result['is_cheater']}, Probability: {result['cheater_probability']:.4f}")
```

### Command Line Usage

```bash
# Train all models and evaluate performance
poetry run python train_evaluate_models.py

# Test the predictor
poetry run python predictor.py

# Test data loading
poetry run python data_loader.py
```

## Technical Documentation

### System Architecture

The system consists of four main components:

1. **Data Loader (`data_loader.py`)**: Handles ARFF file loading, data preprocessing, and feature scaling
2. **Model Trainer (`model_trainer.py`)**: Implements multiple ML algorithms with hyperparameter tuning
3. **Main Pipeline (`train_evaluate_models.py`)**: Orchestrates the complete training and evaluation workflow
4. **Predictor (`predictor.py`)**: Provides interface for making predictions on new data

### Data Specifications

- **Input Format**: ARFF files with 7 numerical features
- **Features**:
  - `Eco_Final_Pred`: Economic prediction metric
  - `Movement_Final_Pred`: Movement pattern metric
  - `Dmg_Final_Pred`: Damage dealing metric
  - `Flash_Final_Pred`: Flash usage metric
  - `Grenade_Final_Pred`: Grenade usage metric
  - `Kill_Final_Pred`: Kill pattern metric
  - `Wf_Final_Pred`: Wallhack/aimbot detection metric
- **Target**: Binary classification (0 = legitimate, 1 = cheater)
- **Training Data**: ~14,400 samples (9 files × ~1,600 samples each)
- **Class Distribution**: Imbalanced dataset (more legitimate players than cheaters)

### Machine Learning Pipeline

#### 1. Data Preprocessing
- **Normalization**: StandardScaler for feature scaling
- **Data Combination**: Merges 9 training files into single dataset
- **Validation Split**: Uses separate validation file for model selection

#### 2. Model Algorithms

The system implements and compares 4 different algorithms:

| Algorithm | Key Parameters | Strengths |
|-----------|---------------|-----------|
| **Random Forest** | n_estimators, max_depth, min_samples_split | Robust, handles imbalanced data, feature importance |
| **Logistic Regression** | C, penalty, solver | Fast, interpretable, good baseline |
| **XGBoost** | n_estimators, max_depth, learning_rate | High performance, handles imbalance |
| **Neural Network** | hidden_layer_sizes, activation, alpha | Complex patterns, non-linear relationships |

#### 3. Hyperparameter Optimization

- **Strategy**: RandomizedSearchCV for complex models, GridSearchCV for simpler ones
- **Scoring Metric**: Recall (prioritizes catching cheaters)
- **Cross-Validation**: 3-fold CV for efficiency
- **Class Balancing**: Uses `class_weight='balanced'` and scale_pos_weight for XGBoost

#### 4. Threshold Optimization

The system optimizes classification thresholds to maximize recall while maintaining reasonable precision:

- **Threshold Range**: 0.1 to 0.9 (step 0.05)
- **Optimization Criterion**: 0.7 × Recall + 0.3 × F1-score
- **Minimum Precision**: 0.3 (prevents excessive false positives)

### Feature Engineering

The current implementation uses minimal feature engineering to maintain simplicity and interpretability:

- **Scaling**: StandardScaler normalization
- **No Feature Selection**: All 7 features are used (can be extended)
- **No Feature Interactions**: Linear relationships only

**Potential Extensions**:
- Polynomial features for capturing interactions
- Feature selection using SelectKBest or RFE
- Domain-specific feature combinations

### Model Evaluation

#### Metrics Used
- **Primary**: Recall (minimize false negatives)
- **Secondary**: Precision, F1-score, Accuracy
- **Probability**: AUC-ROC for ranking quality

#### Evaluation Strategy
1. **Training**: 9 combined training files
2. **Validation**: Separate validation file for model selection and threshold optimization
3. **Testing**: Final evaluation on held-out test set

### Performance Benchmarks

*Note: Actual performance will be displayed after running `main.py`*

Expected performance characteristics:
- **Recall**: >0.85 (primary objective - catch most cheaters)
- **Precision**: >0.30 (avoid too many false positives)
- **F1-Score**: Balanced metric considering both precision and recall
- **AUC**: >0.80 (good ranking ability)

### Model Persistence

Trained models are saved in the `models/` directory:

- `{model_name}_model.joblib`: Individual model files
- `best_model.joblib`: Best performing model with metadata
- `training_results.json`: Complete training results and analysis
- `scaler.joblib`: Feature scaler for preprocessing

### Real-Time Deployment Considerations

The system is designed for real-time gameplay integration:

- **Latency**: Models optimized for fast prediction (<10ms typical)
- **Memory**: Lightweight models suitable for game servers
- **Scalability**: Batch prediction support for multiple players
- **Threshold Tuning**: Adjustable sensitivity based on game requirements

## Usage Examples

### Training New Models

```python
from data_loader import DataLoader
from model_trainer import ModelTrainer

# Load and preprocess data
loader = DataLoader()
train_df = loader.load_training_data()
val_df = loader.load_validation_data()

X_train, y_train = loader.preprocess_data(train_df, fit_scaler=True)
X_val, y_val = loader.preprocess_data(val_df, fit_scaler=False)

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# Save models
trainer.save_models()
```

### Making Predictions

```python
from predictor import AntiCheatPredictor

# Initialize predictor
predictor = AntiCheatPredictor()
predictor.load_model()

# Single player prediction
player_metrics = [5.7e-08, 2.3e-08, 1.2e-15, 3.4e-14, 1.5e-16, 7.6e-08, 2.8e-31]
result = predictor.predict_single(player_metrics)

if result['is_cheater']:
    print(f"⚠️  Potential cheater detected! Confidence: {result['cheater_probability']:.2%}")
else:
    print(f"✅ Player appears legitimate. Confidence: {1-result['cheater_probability']:.2%}")

# Batch predictions for multiple players
batch_metrics = [
    [5.7e-08, 2.3e-08, 1.2e-15, 3.4e-14, 1.5e-16, 7.6e-08, 2.8e-31],
    [5.7e-08, 2.3e-08, 4.5e-12, 9.2e-16, 1.6e-18, 7.6e-08, 2.6e-26]
]
results = predictor.predict_batch(batch_metrics)
```

### Model Analysis

```python
# Get model information
model_info = predictor.get_model_info()
print(f"Using model: {model_info['model_name']}")
print(f"Threshold: {model_info['optimal_threshold']:.4f}")

# Get prediction explanation
explanation = predictor.explain_prediction(player_metrics)
print("Top contributing features:")
for feature in explanation['top_contributing_features']:
    print(f"  Feature {feature['feature_index']}: {feature['contribution']:.2e}")
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure ARFF data files are in the correct directory
2. **Memory Issues**: For large datasets, consider batch processing
3. **Poor Performance**: Check class distribution and consider rebalancing techniques
4. **Slow Training**: Reduce hyperparameter search space or use fewer models

### Performance Tuning

- **For Higher Recall**: Lower the classification threshold
- **For Higher Precision**: Raise the classification threshold
- **For Faster Prediction**: Use simpler models (Logistic Regression, Random Forest)
- **For Better Accuracy**: Increase hyperparameter search iterations

### Logging

Training progress and errors are logged to:
- Console output (INFO level)
- `training.log` file (detailed logs)

## Extension Points

### Adding New Models

```python
# In model_trainer.py, add to get_model_configs():
'new_model': {
    'model': YourModelClass(),
    'params': {
        'param1': [value1, value2],
        'param2': [value3, value4]
    }
}
```

### Custom Feature Engineering

```python
# In data_loader.py, extend preprocess_data():
def custom_feature_engineering(self, X):
    # Add polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    return poly.fit_transform(X)
```

### Integration with Game Systems

```python
# Example game integration
class GameAntiCheat:
    def __init__(self):
        self.predictor = AntiCheatPredictor()
        self.predictor.load_model()
    
    def check_player(self, player_id, game_metrics):
        result = self.predictor.predict_single(game_metrics)
        
        if result['is_cheater'] and result['cheater_probability'] > 0.8:
            self.flag_player(player_id, result)
        
        return result
```

## Support and Maintenance

- **Model Retraining**: Recommended monthly with new data
- **Threshold Adjustment**: Monitor false positive/negative rates
- **Performance Monitoring**: Track prediction latency and accuracy
- **Data Quality**: Ensure consistent feature extraction from game events

For technical support or questions about implementation, refer to the code documentation or contact the development team.
