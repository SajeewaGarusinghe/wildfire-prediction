"""
Configuration settings for the Wildfire Risk Prediction System
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(exist_ok=True)

# Data sources
class DataConfig:
    # Study area (Riverside County, CA)
    STUDY_AREA_BOUNDS = {
        'min_lat': 33.3,
        'max_lat': 34.0,
        'min_lon': -117.7,
        'max_lon': -115.2
    }
    
    # Spatial resolution (meters)
    SPATIAL_RESOLUTION = 60
    
    # Temporal parameters
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SATELLITE_DATA_DIR = RAW_DATA_DIR / "satellite"
    WEATHER_DATA_DIR = RAW_DATA_DIR / "weather"
    FIRE_DATA_DIR = RAW_DATA_DIR / "fire_records"
    MODELS_DIR = MODELS_DIR  # Add the missing MODELS_DIR reference
    
    # API configurations
    NOAA_API_KEY = os.getenv('NOAA_API_KEY', '')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
    
    # Google Earth Engine (requires authentication)
    GEE_SERVICE_ACCOUNT = os.getenv('GEE_SERVICE_ACCOUNT', '')

class ModelConfig:
    # Model parameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    
    # Feature engineering
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'wind_speed', 
        'precipitation', 'pressure'
    ]
    
    VEGETATION_FEATURES = [
        'ndvi', 'evi', 'ndmi', 'nbr'
    ]
    
    TOPOGRAPHIC_FEATURES = [
        'elevation', 'slope', 'aspect'
    ]
    
    DERIVED_FEATURES = [
        'temperature_anomaly', 'drought_index', 
        'fire_weather_index', 'fuel_moisture_estimate'
    ]
    
    # Model architectures
    CNN_LSTM_PARAMS = {
        'cnn_filters': [32, 64, 128],
        'cnn_kernel_size': 3,
        'lstm_units': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    }
    
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_SEED
    }

class DatabaseConfig:
    DATABASE_PATH = DATA_DIR / "wildfire_predictions.db"
    
    # Table schemas
    WEATHER_TABLE = "weather_data"
    SATELLITE_TABLE = "satellite_data"
    FIRE_TABLE = "fire_records"
    PREDICTIONS_TABLE = "predictions"
    LOCATIONS_TABLE = "locations"

class APIConfig:
    HOST = "0.0.0.0"
    PORT = 5050
    DEBUG = False  # Set to False for production deployment
    
    # Risk thresholds (more realistic for wildfire prediction)
    HIGH_RISK_THRESHOLD = 0.4  # 40%+ is high risk
    MODERATE_RISK_THRESHOLD = 0.15  # 15%+ is moderate risk
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 60

class LoggingConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "wildfire_system.log"
