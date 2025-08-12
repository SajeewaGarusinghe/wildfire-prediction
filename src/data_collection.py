"""
Data collection module for wildfire risk prediction system.
Handles downloading and processing of satellite, weather, and fire data.
"""
import requests
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import time

from config import DataConfig, DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collects weather data from NOAA and OpenWeatherMap APIs"""
    
    def __init__(self):
        self.noaa_api_key = DataConfig.NOAA_API_KEY
        self.openweather_api_key = DataConfig.OPENWEATHER_API_KEY
        self.base_url_noaa = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        self.base_url_openweather = "http://api.openweathermap.org/data/2.5"
        
    def get_noaa_stations(self, bounds: Dict) -> List[Dict]:
        """Get weather stations within study area"""
        url = f"{self.base_url_noaa}/stations"
        params = {
            'datasetid': 'GHCND',
            'extent': f"{bounds['min_lat']},{bounds['min_lon']},{bounds['max_lat']},{bounds['max_lon']}",
            'limit': 50
        }
        headers = {'token': self.noaa_api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                return response.json().get('results', [])
            else:
                logger.warning(f"NOAA API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching NOAA stations: {e}")
            return []
    
    def collect_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect weather data for the study period"""
        # For demo purposes, generate synthetic weather data
        # In real implementation, use NOAA API calls
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        locations = self._generate_sample_locations()
        
        weather_data = []
        
        for date in date_range:
            for location in locations:
                # Generate realistic weather patterns
                temp_base = 20 + 15 * np.sin(2 * np.pi * date.dayofyear / 365)
                temp_noise = np.random.normal(0, 5)
                
                weather_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'location_id': location['id'],
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'temperature': temp_base + temp_noise,
                    'humidity': max(10, min(90, 60 + np.random.normal(0, 15))),
                    'wind_speed': max(0, np.random.exponential(8)),
                    'precipitation': max(0, np.random.exponential(2)),
                    'pressure': 1013 + np.random.normal(0, 20)
                }
                weather_data.append(weather_record)
        
        df = pd.DataFrame(weather_data)
        logger.info(f"Generated {len(df)} weather records")
        return df
    
    def _generate_sample_locations(self) -> List[Dict]:
        """Generate sample weather station locations"""
        bounds = DataConfig.STUDY_AREA_BOUNDS
        n_stations = 12
        
        locations = []
        for i in range(n_stations):
            lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
            lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
            
            locations.append({
                'id': f'STATION_{i:02d}',
                'lat': lat,
                'lon': lon,
                'name': f'Weather Station {i+1}'
            })
        
        return locations

class SatelliteDataCollector:
    """Collects satellite imagery and calculates vegetation indices"""
    
    def __init__(self):
        self.data_dir = DataConfig.SATELLITE_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_satellite_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect satellite data and calculate vegetation indices"""
        # For demo purposes, generate synthetic satellite data
        # In real implementation, use Google Earth Engine API
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        grid_points = self._generate_grid_points()
        
        satellite_data = []
        
        for date in date_range:
            for point in grid_points:
                # Generate realistic vegetation indices
                seasonal_factor = 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                satellite_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'location_id': point['id'],
                    'latitude': point['lat'],
                    'longitude': point['lon'],
                    'ndvi': max(0, min(1, 0.6 + seasonal_factor + np.random.normal(0, 0.1))),
                    'evi': max(0, min(1, 0.5 + seasonal_factor + np.random.normal(0, 0.1))),
                    'ndmi': max(0, min(1, 0.4 + seasonal_factor + np.random.normal(0, 0.1))),
                    'nbr': max(0, min(1, 0.7 + seasonal_factor + np.random.normal(0, 0.1))),
                    'red_band': np.random.uniform(0.1, 0.3),
                    'nir_band': np.random.uniform(0.4, 0.8),
                    'swir_band': np.random.uniform(0.1, 0.4)
                }
                satellite_data.append(satellite_record)
        
        df = pd.DataFrame(satellite_data)
        logger.info(f"Generated {len(df)} satellite records")
        return df
    
    def _generate_grid_points(self) -> List[Dict]:
        """Generate grid points for satellite data"""
        bounds = DataConfig.STUDY_AREA_BOUNDS
        resolution = 0.01  # Approximately 1km grid
        
        lats = np.arange(bounds['min_lat'], bounds['max_lat'], resolution)
        lons = np.arange(bounds['min_lon'], bounds['max_lon'], resolution)
        
        grid_points = []
        point_id = 0
        
        for lat in lats:
            for lon in lons:
                grid_points.append({
                    'id': f'GRID_{point_id:04d}',
                    'lat': lat,
                    'lon': lon
                })
                point_id += 1
        
        return grid_points

class FireDataCollector:
    """Collects historical fire records"""
    
    def __init__(self):
        self.data_dir = DataConfig.FIRE_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_fire_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect historical fire records"""
        # Generate synthetic fire records based on realistic patterns
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        fire_records = []
        
        # Fire season typically May through October in California
        fire_season_months = [5, 6, 7, 8, 9, 10]
        
        for date in date_range:
            # Higher probability during fire season
            if date.month in fire_season_months:
                fire_probability = 0.02  # 2% chance per day during fire season
            else:
                fire_probability = 0.005  # 0.5% chance per day outside fire season
            
            if np.random.random() < fire_probability:
                # Generate a fire event
                bounds = DataConfig.STUDY_AREA_BOUNDS
                
                fire_record = {
                    'fire_id': f'FIRE_{len(fire_records):04d}',
                    'date': date.strftime('%Y-%m-%d'),
                    'latitude': np.random.uniform(bounds['min_lat'], bounds['max_lat']),
                    'longitude': np.random.uniform(bounds['min_lon'], bounds['max_lon']),
                    'acres_burned': np.random.lognormal(3, 2),  # Log-normal distribution for fire size
                    'cause': np.random.choice(['Lightning', 'Human', 'Equipment', 'Unknown']),
                    'confidence': np.random.uniform(0.7, 1.0)
                }
                fire_records.append(fire_record)
        
        df = pd.DataFrame(fire_records)
        logger.info(f"Generated {len(df)} fire records")
        return df

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self):
        self.db_path = DatabaseConfig.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Weather data table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DatabaseConfig.WEATHER_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                location_id TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                precipitation REAL,
                pressure REAL,
                UNIQUE(date, location_id)
            )
        ''')
        
        # Satellite data table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DatabaseConfig.SATELLITE_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                location_id TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                ndvi REAL,
                evi REAL,
                ndmi REAL,
                nbr REAL,
                red_band REAL,
                nir_band REAL,
                swir_band REAL,
                UNIQUE(date, location_id)
            )
        ''')
        
        # Fire records table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DatabaseConfig.FIRE_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fire_id TEXT UNIQUE NOT NULL,
                date TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                acres_burned REAL,
                cause TEXT,
                confidence REAL
            )
        ''')
        
        # Predictions table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DatabaseConfig.PREDICTIONS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                location_id TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        """Save DataFrame to database table"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.close()
        logger.info(f"Saved {len(df)} records to {table_name}")
    
    def load_data(self, table_name: str, where_clause: str = "") -> pd.DataFrame:
        """Load data from database table"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

def main():
    """Main data collection pipeline"""
    logger.info("Starting data collection pipeline")
    
    # Initialize collectors and database
    weather_collector = WeatherDataCollector()
    satellite_collector = SatelliteDataCollector()
    fire_collector = FireDataCollector()
    db_manager = DatabaseManager()
    
    # Collect data
    start_date = DataConfig.START_DATE
    end_date = DataConfig.END_DATE
    
    logger.info("Collecting weather data...")
    weather_data = weather_collector.collect_weather_data(start_date, end_date)
    db_manager.save_dataframe(weather_data, DatabaseConfig.WEATHER_TABLE)
    
    logger.info("Collecting satellite data...")
    satellite_data = satellite_collector.collect_satellite_data(start_date, end_date)
    db_manager.save_dataframe(satellite_data, DatabaseConfig.SATELLITE_TABLE)
    
    logger.info("Collecting fire data...")
    fire_data = fire_collector.collect_fire_data(start_date, end_date)
    db_manager.save_dataframe(fire_data, DatabaseConfig.FIRE_TABLE)
    
    logger.info("Data collection pipeline completed successfully")

if __name__ == "__main__":
    main()
