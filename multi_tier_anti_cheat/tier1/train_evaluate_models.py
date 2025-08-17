"""
Main execution script for anti-cheat ML detection system.
Orchestrates the complete pipeline from data loading to model evaluation.
"""

import os
import json
import logging
from datetime import datetime
from data_loader import DataLoader
from model_trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("Starting Anti-Cheat ML Detection System")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        loader = DataLoader()
        trainer = ModelTrainer()
        
        # Step 1: Load and analyze data
        logger.info("Step 1: Loading and analyzing data...")
        train_df = loader.load_training_data()
        val_df = loader.load_validation_data()
        test_df = loader.load_test_data()
        
        # Analyze data distributions
        train_analysis = loader.analyze_data_distribution(train_df)
        val_analysis = loader.analyze_data_distribution(val_df)
        test_analysis = loader.analyze_data_distribution(test_df)
        
        logger.info(f"Training data: {train_analysis['total_samples']} samples")
        logger.info(f"Class distribution: {train_analysis['class_distribution']}")
        logger.info(f"Class balance ratio: {train_analysis['class_balance_ratio']:.4f}")
        
        # Step 2: Preprocess data
        logger.info("\nStep 2: Preprocessing data...")
        X_train, y_train = loader.preprocess_data(train_df, fit_scaler=True)
        X_val, y_val = loader.preprocess_data(val_df, fit_scaler=False)
        X_test, y_test = loader.preprocess_data(test_df, fit_scaler=False)
        
        # Step 3: Train all models
        logger.info("\nStep 3: Training models...")
        training_results = trainer.train_all_models(X_train, y_train, X_val, y_val)
        
        # Step 4: Evaluate best model on test set
        logger.info("\nStep 4: Evaluating best model on test set...")
        test_results = trainer.evaluate_on_test(X_test, y_test)
        
        # Step 5: Feature importance analysis
        logger.info("\nStep 5: Analyzing feature importance...")
        feature_names = loader.get_feature_names()
        importance_analysis = trainer.get_feature_importance_analysis(feature_names)
        
        # Step 6: Save models and results
        logger.info("\nStep 6: Saving models and results...")
        trainer.save_models()
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'data_analysis': {
                'training': train_analysis,
                'validation': val_analysis,
                'test': test_analysis
            },
            'training_results': training_results,
            'test_results': test_results,
            'feature_importance': importance_analysis,
            'best_model': trainer.best_model_name
        }
        
        # Save results to JSON
        results_path = os.path.join(trainer.models_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"Best Model: {trainer.best_model_name}")
        print(f"Optimal Threshold: {test_results['optimal_threshold']:.4f}")
        print("\nTest Performance:")
        print(f"  Accuracy:  {test_results['test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {test_results['test_metrics']['precision']:.4f}")
        print(f"  Recall:    {test_results['test_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {test_results['test_metrics']['f1']:.4f}")
        print(f"  AUC:       {test_results['test_metrics']['auc']:.4f}")
        
        if importance_analysis:
            print(f"\nMost Important Features:")
            for i, feature in enumerate(importance_analysis.get('most_important_features', []), 1):
                print(f"  {i}. {feature}")
        
        print(f"\nModels saved to: {trainer.models_dir}")
        print(f"Results saved to: {results_path}")
        print("=" * 80)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
