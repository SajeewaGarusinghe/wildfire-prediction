"""
Feature engineering module for wildfire risk prediction.
Processes raw data and creates features for machine learning models.
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, List, Tuple, Optional

from config import DataConfig, DatabaseConfig, ModelConfig

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self):
        self.db_path = DatabaseConfig.DATABASE_PATH
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_raw_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load raw data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load weather data
        weather_query = f"""
            SELECT * FROM {DatabaseConfig.WEATHER_TABLE}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Load satellite data
        satellite_query = f"""
            SELECT * FROM {DatabaseConfig.SATELLITE_TABLE}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        satellite_df = pd.read_sql_query(satellite_query, conn)
        
        # Load fire data
        fire_query = f"""
            SELECT * FROM {DatabaseConfig.FIRE_TABLE}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        fire_df = pd.read_sql_query(fire_query, conn)
        
        conn.close()
        
        return {
            'weather': weather_df,
            'satellite': satellite_df,
            'fire': fire_df
        }
    
    def create_spatial_grid(self, resolution: float = 0.01) -> pd.DataFrame:
        """Create spatial grid for consistent location mapping"""
        bounds = DataConfig.STUDY_AREA_BOUNDS
        
        lats = np.arange(bounds['min_lat'], bounds['max_lat'], resolution)
        lons = np.arange(bounds['min_lon'], bounds['max_lon'], resolution)
        
        grid_points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                grid_points.append({
                    'grid_id': f'GRID_{i:03d}_{j:03d}',
                    'latitude': lat,
                    'longitude': lon,
                    'grid_x': j,
                    'grid_y': i
                })
        
        return pd.DataFrame(grid_points)
    
    def spatial_interpolation(self, df: pd.DataFrame, target_grid: pd.DataFrame, 
                            value_columns: List[str]) -> pd.DataFrame:
        """Interpolate point data to regular grid"""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            logger.warning("SciPy not available, using simple nearest neighbor interpolation")
            # Simple fallback without scipy
            return self._simple_spatial_interpolation(df, target_grid, value_columns)
        
        results = []
        
        for date in df['date'].unique():
            date_data = df[df['date'] == date].copy()
            
            if len(date_data) == 0:
                continue
            
            # Create KDTree for spatial interpolation
            coords = date_data[['latitude', 'longitude']].values
            tree = KDTree(coords)
            
            # Interpolate to grid points
            for _, grid_point in target_grid.iterrows():
                grid_coords = np.array([[grid_point['latitude'], grid_point['longitude']]])
                
                # Find nearest neighbors
                distances, indices = tree.query(grid_coords, k=min(5, len(date_data)))
                distances = distances[0]
                indices = indices[0]
                
                # Inverse distance weighting
                weights = 1 / (distances + 1e-10)  # Avoid division by zero
                weights = weights / np.sum(weights)
                
                # Interpolate values
                interpolated_values = {}
                for col in value_columns:
                    values = date_data.iloc[indices][col].values
                    interpolated_values[col] = np.sum(values * weights)
                
                # Create record
                record = {
                    'date': date,
                    'grid_id': grid_point['grid_id'],
                    'latitude': grid_point['latitude'],
                    'longitude': grid_point['longitude'],
                    **interpolated_values
                }
                results.append(record)
        
        return pd.DataFrame(results)
    
    def _simple_spatial_interpolation(self, df: pd.DataFrame, target_grid: pd.DataFrame, 
                                    value_columns: List[str]) -> pd.DataFrame:
        """Simple spatial interpolation without scipy dependency"""
        results = []
        
        for date in df['date'].unique():
            date_data = df[df['date'] == date].copy()
            
            if len(date_data) == 0:
                continue
            
            # For each grid point, find nearest data point using simple distance
            for _, grid_point in target_grid.iterrows():
                grid_lat, grid_lon = grid_point['latitude'], grid_point['longitude']
                
                # Calculate simple Euclidean distances
                distances = np.sqrt(
                    (date_data['latitude'] - grid_lat)**2 + 
                    (date_data['longitude'] - grid_lon)**2
                )
                
                # Find nearest point
                nearest_idx = distances.idxmin()
                nearest_data = date_data.loc[nearest_idx]
                
                # Create interpolated record
                record = {
                    'date': date,
                    'grid_id': grid_point['grid_id'],
                    'latitude': grid_point['latitude'],
                    'longitude': grid_point['longitude']
                }
                
                # Copy values from nearest point
                for col in value_columns:
                    record[col] = nearest_data[col]
                
                results.append(record)
        
        return pd.DataFrame(results)
    
    def calculate_temporal_features(self, df: pd.DataFrame, 
                                  value_columns: List[str], 
                                  window_sizes: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Calculate temporal features (moving averages, trends)"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['grid_id', 'date'])
        
        for col in value_columns:
            for window in window_sizes:
                # Moving average
                df[f'{col}_ma_{window}d'] = df.groupby('grid_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Moving standard deviation
                df[f'{col}_std_{window}d'] = df.groupby('grid_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                
                # Anomaly (current value vs moving average)
                df[f'{col}_anomaly_{window}d'] = df[col] - df[f'{col}_ma_{window}d']
        
        return df
    
    def calculate_derived_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from multiple data sources"""
        df = merged_df.copy()
        
        # Fire Weather Index (simplified)
        df['fire_weather_index'] = (
            (df['temperature'] * df['wind_speed']) / 
            (df['humidity'] + 1e-10)  # Avoid division by zero
        )
        
        # Drought index (precipitation deficit)
        df['drought_index'] = (
            df['precipitation_ma_30d'] / 
            (df['precipitation_ma_30d'].mean() + 1e-10)
        )
        
        # Fuel moisture estimate (based on NDVI and weather)
        df['fuel_moisture_estimate'] = (
            df['ndvi'] * df['humidity'] / 
            (df['temperature'] + 1e-10)
        )
        
        # Vegetation stress indicator
        df['vegetation_stress'] = (
            df['ndvi_ma_30d'] - df['ndvi']
        ) / (df['ndvi_ma_30d'] + 1e-10)
        
        # Seasonal features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })
        
        # Cyclic encoding for temporal features
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def create_fire_labels(self, merged_df: pd.DataFrame, fire_df: pd.DataFrame,
                          spatial_buffer: float = 0.01, temporal_buffer: int = 7) -> pd.DataFrame:
        """Create binary fire occurrence labels for supervised learning"""
        df = merged_df.copy()
        df['fire_occurred'] = 0
        df['days_to_fire'] = np.inf
        
        df['date'] = pd.to_datetime(df['date'])
        fire_df['date'] = pd.to_datetime(fire_df['date'])
        
        for _, fire in fire_df.iterrows():
            # Spatial buffer
            lat_min = fire['latitude'] - spatial_buffer
            lat_max = fire['latitude'] + spatial_buffer
            lon_min = fire['longitude'] - spatial_buffer
            lon_max = fire['longitude'] + spatial_buffer
            
            # Temporal buffer
            date_min = fire['date'] - timedelta(days=temporal_buffer)
            date_max = fire['date'] + timedelta(days=temporal_buffer)
            
            # Find matching records
            spatial_mask = (
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
            )
            temporal_mask = (df['date'] >= date_min) & (df['date'] <= date_max)
            
            fire_mask = spatial_mask & temporal_mask
            df.loc[fire_mask, 'fire_occurred'] = 1
            
            # Calculate days to fire for non-fire instances
            before_fire_mask = spatial_mask & (df['date'] < fire['date'])
            days_diff = (fire['date'] - df.loc[before_fire_mask, 'date']).dt.days
            
            current_days = df.loc[before_fire_mask, 'days_to_fire']
            df.loc[before_fire_mask, 'days_to_fire'] = np.minimum(current_days, days_diff)
        
        # Create risk score based on proximity to fires
        df['fire_risk_score'] = np.where(
            df['fire_occurred'] == 1, 
            1.0,
            np.where(
                df['days_to_fire'] <= 30,
                1.0 - (df['days_to_fire'] / 30.0),
                0.0
            )
        )
        
        return df
    
    def prepare_features(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[str]]:
        """Main feature preparation pipeline"""
        logger.info("Loading raw data...")
        raw_data = self.load_raw_data(start_date, end_date)
        
        logger.info("Creating spatial grid...")
        grid = self.create_spatial_grid()
        
        logger.info("Interpolating weather data to grid...")
        weather_features = ModelConfig.WEATHER_FEATURES
        weather_grid = self.spatial_interpolation(
            raw_data['weather'], grid, weather_features
        )
        
        logger.info("Interpolating satellite data to grid...")
        vegetation_features = ModelConfig.VEGETATION_FEATURES
        satellite_grid = self.spatial_interpolation(
            raw_data['satellite'], grid, vegetation_features
        )
        
        logger.info("Merging datasets...")
        merged_df = pd.merge(
            weather_grid, satellite_grid,
            on=['date', 'grid_id', 'latitude', 'longitude'],
            how='inner'
        )
        
        logger.info("Calculating temporal features...")
        merged_df = self.calculate_temporal_features(
            merged_df, weather_features + vegetation_features
        )
        
        logger.info("Calculating derived features...")
        merged_df = self.calculate_derived_features(merged_df)
        
        logger.info("Creating fire labels...")
        final_df = self.create_fire_labels(merged_df, raw_data['fire'])
        
        # Define feature columns
        feature_columns = (
            weather_features + vegetation_features + 
            ModelConfig.DERIVED_FEATURES +
            ['day_of_year_sin', 'day_of_year_cos', 'season'] +
            [col for col in final_df.columns if any(
                suffix in col for suffix in ['_ma_', '_std_', '_anomaly_']
            )]
        )
        
        # Filter to existing columns
        feature_columns = [col for col in feature_columns if col in final_df.columns]
        
        logger.info(f"Feature engineering completed. Generated {len(feature_columns)} features")
        
        return final_df, feature_columns
    
    def preprocess_features(self, df: pd.DataFrame, feature_columns: List[str],
                          fit_scalers: bool = True) -> np.ndarray:
        """Preprocess features for model training"""
        X = df[feature_columns].copy()
        
        # Handle missing values
        if fit_scalers:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        
        # Scale features
        if fit_scalers:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled

def main():
    """Test feature engineering pipeline"""
    engineer = FeatureEngineer()
    
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    features_df, feature_columns = engineer.prepare_features(start_date, end_date)
    
    print(f"Generated features dataset with shape: {features_df.shape}")
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Fire occurrence rate: {features_df['fire_occurred'].mean():.4f}")
    
    # Save processed features
    features_df.to_csv(DataConfig.PROCESSED_DATA_DIR / "features_2020.csv", index=False)
    
    with open(DataConfig.PROCESSED_DATA_DIR / "feature_columns.txt", 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")

if __name__ == "__main__":
    main()
