#!/usr/bin/env python3
"""
Simple pipeline runner that creates training data and trains models
without complex spatial interpolation or UCI dataset processing.
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

def run_data_collection():
    """Run simple data collection"""
    logger.info("="*50)
    logger.info("STARTING SIMPLE DATA COLLECTION")
    logger.info("="*50)
    
    try:
        from src.data_collection_simple import main as collect_data
        collect_data()
        logger.info("Simple data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_feature_engineering():
    """Run simple feature engineering"""
    logger.info("="*50)
    logger.info("STARTING SIMPLE FEATURE ENGINEERING")
    logger.info("="*50)
    
    try:
        from src.feature_engineering_simple import main as engineer_features
        engineer_features()
        logger.info("Simple feature engineering completed successfully")
        return True
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_model_training():
    """Run simple model training"""
    logger.info("="*50)
    logger.info("STARTING SIMPLE MODEL TRAINING")
    logger.info("="*50)
    
    try:
        from src.models_simple import main as train_models
        train_models()
        logger.info("Simple model training completed successfully")
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
        
        # Check processed data
        processed_dir = Path('data/processed')
        if processed_dir.exists():
            processed_files = list(processed_dir.glob('*.csv'))
            logger.info(f"✅ Processed data directory created")
            logger.info(f"   Processed files: {len(processed_files)}")
            for proc_file in processed_files:
                logger.info(f"   - {proc_file.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking results: {e}")
        return False

def main():
    """Run the complete simple pipeline"""
    logger.info("="*60)
    logger.info("WILDFIRE PREDICTION - SIMPLE PIPELINE")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Step 1: Clear database
    clear_database()
    
    # Step 2: Data collection
    if not run_data_collection():
        logger.error("Pipeline failed at data collection stage")
        sys.exit(1)
    
    # Step 3: Feature engineering
    if not run_feature_engineering():
        logger.error("Pipeline failed at feature engineering stage")
        sys.exit(1)
    
    # Step 4: Model training
    if not run_model_training():
        logger.error("Pipeline failed at model training stage")
        sys.exit(1)
    
    # Step 5: Check results
    if not check_results():
        logger.error("Pipeline completed but results validation failed")
        sys.exit(1)
    
    # Success!
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("="*60)
    logger.info("✅ Ready for predictions!")
    logger.info("   - Database with realistic training data created")
    logger.info("   - Multiple ML models trained and saved")
    logger.info("   - Web application can now make real predictions")
    logger.info("="*60)

if __name__ == "__main__":
    main()
