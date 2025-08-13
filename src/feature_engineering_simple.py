"""
Simplified feature engineering that works directly with training_data table
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

from config import DatabaseConfig

logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simple feature engineering for wildfire prediction"""
    
    def __init__(self):
        self.db_path = DatabaseConfig.DATABASE_PATH
        self.scaler = StandardScaler()
    
    def load_training_data(self) -> pd.DataFrame:
        """Load preprocessed training data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM training_data"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} training records")
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better prediction"""
        df = df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Season (categorical)
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })
        
        # Fire season indicator
        df['fire_season'] = df['month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10] else 0)
        
        # Cyclic encoding for temporal features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weather-based derived features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['wind_temp_index'] = df['wind_speed'] * df['temperature'] / 100
        df['dryness_index'] = (100 - df['humidity']) * df['temperature'] / 100
        df['fire_weather_index'] = (
            df['temperature'] * 0.3 + 
            (100 - df['humidity']) * 0.3 + 
            df['wind_speed'] * 0.2 + 
            (1 / (df['precipitation'] + 0.1)) * 0.2
        )
        
        # Vegetation-based features
        df['vegetation_stress'] = (1 - df['ndvi']) * (1 - df['ndmi'])
        df['fuel_moisture'] = df['ndmi'] * df['humidity'] / 100
        df['vegetation_health'] = (df['ndvi'] + df['evi']) / 2
        
        # Combined risk indicators
        df['drought_stress'] = (
            (100 - df['humidity']) / 100 * 0.4 +
            (1 - df['ndmi']) * 0.3 +
            (1 / (df['precipitation'] + 0.1)) * 0.3
        )
        
        df['ignition_potential'] = (
            df['temperature'] / 40 * 0.3 +
            df['wind_speed'] / 30 * 0.3 +
            (100 - df['humidity']) / 100 * 0.4
        )
        
        logger.info(f"Added derived features. Dataset shape: {df.shape}")
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based features"""
        df = df.copy()
        
        # Geographic features
        df['lat_normalized'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
        df['lon_normalized'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
        
        # Distance from coast (approximate)
        coast_lon = -120.0  # Approximate California coast longitude
        df['distance_from_coast'] = np.abs(df['longitude'] - coast_lon)
        
        # Elevation proxy (rough approximation based on latitude)
        df['elevation_proxy'] = df['latitude'] * 100  # Very rough approximation
        
        return df
    
    def prepare_features(self) -> tuple:
        """Main feature preparation pipeline"""
        logger.info("Starting feature preparation...")
        
        # Load data
        df = self.load_training_data()
        
        if len(df) == 0:
            logger.error("No training data found!")
            return None, None, None, None
        
        # Add derived features
        df = self.add_derived_features(df)
        df = self.create_location_features(df)
        
        # Define feature columns (exclude target and metadata)
        feature_columns = [
            # Basic weather features
            'temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure',
            # Vegetation features
            'ndvi', 'evi', 'ndmi', 'nbr',
            # Temporal features
            'day_of_year', 'month', 'season', 'fire_season',
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            # Derived weather features
            'temp_humidity_ratio', 'wind_temp_index', 'dryness_index', 'fire_weather_index',
            # Derived vegetation features
            'vegetation_stress', 'fuel_moisture', 'vegetation_health',
            # Combined risk features
            'drought_stress', 'ignition_potential',
            # Location features
            'lat_normalized', 'lon_normalized', 'distance_from_coast', 'elevation_proxy'
        ]
        
        # Filter to existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['fire_occurred'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        logger.info(f"Feature preparation completed:")
        logger.info(f"  Dataset shape: {X.shape}")
        logger.info(f"  Feature columns: {len(feature_columns)}")
        logger.info(f"  Fire occurrence rate: {y.mean():.4f}")
        logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns, df
    
    def get_feature_importance_names(self) -> list:
        """Get human-readable feature names for importance plotting"""
        return [
            'Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Pressure',
            'NDVI', 'EVI', 'NDMI', 'NBR',
            'Day of Year', 'Month', 'Season', 'Fire Season',
            'Day Sin', 'Day Cos', 'Month Sin', 'Month Cos',
            'Temp/Humidity Ratio', 'Wind-Temp Index', 'Dryness Index', 'Fire Weather Index',
            'Vegetation Stress', 'Fuel Moisture', 'Vegetation Health',
            'Drought Stress', 'Ignition Potential',
            'Latitude (norm)', 'Longitude (norm)', 'Distance from Coast', 'Elevation Proxy'
        ]

def main():
    """Test feature engineering"""
    engineer = SimpleFeatureEngineer()
    
    X, y, feature_columns, df = engineer.prepare_features()
    
    if X is not None:
        print(f"Successfully prepared features:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Feature columns: {feature_columns}")
        print(f"\nFeature statistics:")
        print(X.describe())
        
        # Save processed features
        from pathlib import Path
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        X.to_csv(processed_dir / "features.csv", index=False)
        y.to_csv(processed_dir / "targets.csv", index=False)
        
        # Save feature names
        with open(processed_dir / "feature_names.txt", 'w') as f:
            for col in feature_columns:
                f.write(f"{col}\n")
        
        print(f"\nSaved processed data to {processed_dir}")
    
    else:
        print("Feature preparation failed!")

if __name__ == "__main__":
    main()
