#!/usr/bin/env python3
"""
Simple script to clear database and run pipeline with Kaggle/real datasets
"""
import sqlite3
import os
from pathlib import Path
import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
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

def run_pipeline():
    """Run the pipeline with Kaggle datasets"""
    logger.info("Starting pipeline with real datasets...")
    
    try:
        # Run the pipeline
        result = subprocess.run([
            sys.executable, 'run_pipeline.py', 
            '--stage', 'full',
            '--use-kaggle'
        ], check=True, capture_output=True, text=True)
        
        logger.info("Pipeline completed successfully!")
        logger.info("Output:")
        print(result.stdout)
        
        if result.stderr:
            logger.warning("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed with exit code {e.returncode}")
        logger.error("STDOUT:")
        print(e.stdout)
        logger.error("STDERR:")
        print(e.stderr)
        return False
    
    return True

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("WILDFIRE PREDICTION - KAGGLE DATASET PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Clear database
    clear_database()
    
    # Step 2: Run pipeline
    success = run_pipeline()
    
    if success:
        logger.info("=" * 60)
        logger.info("SUCCESS! Pipeline completed with real datasets")
        logger.info("The web application can now make predictions using trained models")
        logger.info("=" * 60)
    else:
        logger.error("Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
