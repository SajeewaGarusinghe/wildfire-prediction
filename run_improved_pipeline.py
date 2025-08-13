#!/usr/bin/env python3
"""
Improved pipeline runner with better model training
"""
import sqlite3
import os
from pathlib import Path
import logging
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_database():
    """Clear the existing database"""
    db_path = Path('data/wildfire_predictions.db')
    
    if db_path.exists():
        logger.info("Removing existing database...")
        os.remove(db_path)
        logger.info("Database cleared successfully")
    else:
        logger.info("No existing database found")
    
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Ready for fresh data collection")

def clear_models():
    """Clear existing models to force retraining"""
    models_dir = Path('models')
    if models_dir.exists():
        # Clear individual files instead of removing directory
        import glob
        model_files = glob.glob(str(models_dir / '*'))
        for file_path in model_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
        logger.info("Cleared existing model files")
    models_dir.mkdir(parents=True, exist_ok=True)

def run_data_collection():
    """Run simple data collection"""
    logger.info("="*50)
    logger.info("STARTING DATA COLLECTION")
    logger.info("="*50)
    
    try:
        from src.data_collection_simple import main as collect_data
        collect_data()
        logger.info("Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_feature_engineering():
    """Run simple feature engineering"""
    logger.info("="*50)
    logger.info("STARTING FEATURE ENGINEERING")
    logger.info("="*50)
    
    try:
        from src.feature_engineering_simple import main as engineer_features
        engineer_features()
        logger.info("Feature engineering completed successfully")
        return True
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_improved_model_training():
    """Run improved model training"""
    logger.info("="*50)
    logger.info("STARTING IMPROVED MODEL TRAINING")
    logger.info("="*50)
    
    try:
        from src.models_improved import main as train_models
        results = train_models()
        logger.info("Improved model training completed successfully")
        
        # Log final results
        logger.info("🎯 FINAL MODEL PERFORMANCE:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_results():
    """Check the results of the pipeline"""
    logger.info("="*50)
    logger.info("CHECKING PIPELINE RESULTS")
    logger.info("="*50)
    
    try:
        # Check database
        db_path = Path('data/wildfire_predictions.db')
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            
            # Check training data
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_data")
            training_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM training_data WHERE fire_occurred = 1")
            fire_count = cursor.fetchone()[0]
            
            fire_rate = fire_count / training_count if training_count > 0 else 0
            
            logger.info(f"✅ Database created successfully")
            logger.info(f"   Training records: {training_count:,}")
            logger.info(f"   Fire occurrences: {fire_count:,}")
            logger.info(f"   Fire rate: {fire_rate:.4f}")
            
            conn.close()
        else:
            logger.error("❌ Database not found")
            return False
        
        # Check models
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.h5'))
            logger.info(f"✅ Models directory created")
            logger.info(f"   Model files: {len(model_files)}")
            for model_file in model_files:
                logger.info(f"   - {model_file.name}")
        else:
            logger.error("❌ Models directory not found")
            return False
        
        # Test model loading
        try:
            import pickle
            
            # Test Random Forest
            with open(models_dir / "random_forest.pkl", 'rb') as f:
                rf_model = pickle.load(f)
            logger.info(f"✅ Random Forest loaded successfully")
            
            # Test scaler
            with open(models_dir / "scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"✅ Feature scaler loaded successfully")
            
            # Test neural network if exists
            nn_path = models_dir / "neural_network.h5"
            if nn_path.exists():
                try:
                    import tensorflow as tf
                    nn_model = tf.keras.models.load_model(nn_path)
                    
                    # Quick test prediction
                    import numpy as np
                    test_input = np.random.random((1, 25))  # 25 features
                    pred = nn_model.predict(test_input, verbose=0)
                    logger.info(f"✅ Neural Network loaded successfully (test pred: {pred[0][0]:.6f})")
                except Exception as e:
                    logger.warning(f"⚠️ Neural Network loading issue: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error testing model loading: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking results: {e}")
        return False

def main():
    """Run the complete improved pipeline"""
    logger.info("="*70)
    logger.info("WILDFIRE PREDICTION - IMPROVED PIPELINE")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Step 1: Clear everything for fresh training
    clear_database()
    clear_models()
    
    # Step 2: Data collection
    if not run_data_collection():
        logger.error("Pipeline failed at data collection stage")
        sys.exit(1)
    
    # Step 3: Feature engineering
    if not run_feature_engineering():
        logger.error("Pipeline failed at feature engineering stage")
        sys.exit(1)
    
    # Step 4: Improved model training
    if not run_improved_model_training():
        logger.error("Pipeline failed at model training stage")
        sys.exit(1)
    
    # Step 5: Check results
    if not check_results():
        logger.error("Pipeline completed but results validation failed")
        sys.exit(1)
    
    # Success!
    total_time = time.time() - start_time
    logger.info("="*70)
    logger.info("🎉 IMPROVED PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("="*70)
    logger.info("🔥 IMPROVEMENTS:")
    logger.info("   ✅ Better neural network architecture with batch normalization")
    logger.info("   ✅ Class imbalance handling with balanced sampling")
    logger.info("   ✅ Improved hyperparameters for all models")
    logger.info("   ✅ Added Gradient Boosting model")
    logger.info("   ✅ Enhanced regularization and dropout")
    logger.info("   ✅ Better training callbacks and early stopping")
    logger.info("="*70)

if __name__ == "__main__":
    main()
