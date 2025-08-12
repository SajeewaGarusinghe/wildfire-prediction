#!/usr/bin/env python3
"""
Main pipeline runner for the Wildfire Risk Prediction System.
Orchestrates data collection, feature engineering, model training, and evaluation.
"""
import argparse
import logging
import sys
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import pipeline modules
from src.data_collection import main as collect_data
from src.feature_engineering import main as engineer_features
from src.models import main as train_models

def run_data_collection():
    """Run data collection pipeline"""
    logger.info("="*50)
    logger.info("STARTING DATA COLLECTION PIPELINE")
    logger.info("="*50)
    
    start_time = time.time()
    
    try:
        collect_data()
        elapsed_time = time.time() - start_time
        logger.info(f"Data collection completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def run_feature_engineering():
    """Run feature engineering pipeline"""
    logger.info("="*50)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("="*50)
    
    start_time = time.time()
    
    try:
        engineer_features()
        elapsed_time = time.time() - start_time
        logger.info(f"Feature engineering completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return False

def run_model_training():
    """Run model training pipeline"""
    logger.info("="*50)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("="*50)
    
    start_time = time.time()
    
    try:
        train_models()
        elapsed_time = time.time() - start_time
        logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline"""
    logger.info("="*60)
    logger.info("STARTING FULL WILDFIRE PREDICTION PIPELINE")
    logger.info("="*60)
    
    total_start_time = time.time()
    
    # Step 1: Data Collection
    if not run_data_collection():
        logger.error("Pipeline failed at data collection stage")
        return False
    
    # Step 2: Feature Engineering
    if not run_feature_engineering():
        logger.error("Pipeline failed at feature engineering stage")
        return False
    
    # Step 3: Model Training
    if not run_model_training():
        logger.error("Pipeline failed at model training stage")
        return False
    
    # Pipeline completed successfully
    total_elapsed_time = time.time() - total_start_time
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total execution time: {total_elapsed_time:.2f} seconds")
    logger.info("="*60)
    
    return True

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if required directories exist
    required_dirs = ['data', 'models', 'logs', 'src']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        logger.info("Creating missing directories...")
        for dir_name in missing_dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
    
    # Check if Python packages are installed
    try:
        import pandas
        import numpy
        import tensorflow
        import sklearn
        import flask
        logger.info("All required packages are available")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Please install requirements: pip install -r requirements.txt")
        return False
    
    logger.info("Prerequisites check passed")
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Wildfire Risk Prediction System Pipeline"
    )
    
    parser.add_argument(
        '--stage', 
        choices=['data', 'features', 'models', 'full'],
        default='full',
        help='Pipeline stage to run (default: full)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip prerequisite checks'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            logger.error("Prerequisites check failed. Exiting.")
            sys.exit(1)
    
    # Run selected pipeline stage
    success = False
    
    if args.stage == 'data':
        success = run_data_collection()
    elif args.stage == 'features':
        success = run_feature_engineering()
    elif args.stage == 'models':
        success = run_model_training()
    elif args.stage == 'full':
        success = run_full_pipeline()
    
    # Exit with appropriate code
    if success:
        logger.info("Pipeline execution completed successfully")
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
