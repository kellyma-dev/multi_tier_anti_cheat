import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, List
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of ARFF data files."""
    current_dir = Path(__file__).resolve().parent

    def __init__(self):
        self.data_dir = os.path.join(DataLoader.current_dir, "../../data/RevPOV/dataset/arff/")
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_arff_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a single ARFF file and convert to DataFrame.
        """

        try:
            data, meta = arff.loadarff(filepath)
            df = pd.DataFrame(data)
            
            # Convert bytes to string if necessary and handle target column
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            # Convert target column to numeric
            if 'isCheater' in df.columns:
                df['isCheater'] = df['isCheater'].astype(int)
                
            logger.info(f"Loaded {filepath}: {len(df)} samples, {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load and combine all 9 training files.
        """

        train_files = [f"final_train_{i}.arff" for i in range(1, 10)]
        combined_data = []
        
        for filename in train_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                df = self.load_arff_file(filepath)
                combined_data.append(df)
            else:
                logger.warning(f"Training file not found: {filepath}")
        
        if not combined_data:
            raise FileNotFoundError("No training files found!")
        
        # Combine all training data
        train_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined training data: {len(train_df)} samples")
        
        # Store feature names
        self.feature_names = [col for col in train_df.columns if col != 'isCheater']
        
        return train_df
    
    def load_validation_data(self) -> pd.DataFrame:
        """
        Load validation data.
        """
        val_file = os.path.join(self.data_dir, "final_val.arff")
        return self.load_arff_file(val_file)
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data.
        """
        test_file = os.path.join(self.data_dir, "final_test.arff")
        return self.load_arff_file(test_file)
    
    def preprocess_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data by separating features and target, and scaling features.
        Returns:
            Tuple of (features, target)
        """
        # Separate features and target
        X = df.drop('isCheater', axis=1).values
        y = df['isCheater'].values
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted scaler on training data")
        else:
            X_scaled = self.scaler.transform(X)
        
        logger.info(f"Preprocessed data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X_scaled, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names if self.feature_names else []
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> dict:
        """
        Analyze data distribution and basic statistics.
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic info
        analysis['total_samples'] = len(df)
        analysis['num_features'] = len(df.columns) - 1  # Excluding target
        
        # Class distribution
        class_counts = df['isCheater'].value_counts()
        analysis['class_distribution'] = class_counts.to_dict()
        analysis['class_balance_ratio'] = class_counts.min() / class_counts.max()
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col != 'isCheater']
        analysis['feature_stats'] = df[feature_cols].describe().to_dict()
        
        # Check for missing values
        analysis['missing_values'] = df.isnull().sum().to_dict()
        
        return analysis


def main():
    """Test the data loader."""
    loader = DataLoader()
    
    # Load and analyze training data
    print("Loading training data...")
    train_df = loader.load_training_data()
    train_analysis = loader.analyze_data_distribution(train_df)
    print(f"Training data analysis: {train_analysis}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_df = loader.load_validation_data()
    val_analysis = loader.analyze_data_distribution(val_df)
    print(f"Validation data analysis: {val_analysis}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train = loader.preprocess_data(train_df, fit_scaler=True)
    X_val, y_val = loader.preprocess_data(val_df, fit_scaler=False)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    print(f"Feature names: {loader.get_feature_names()}")

    # Load test data
    print("\nLoading test data...")
    val_df = loader.load_test_data()
    val_analysis = loader.analyze_data_distribution(val_df)
    print(f"Test data analysis: {val_analysis}")

if __name__ == "__main__":
    main()
