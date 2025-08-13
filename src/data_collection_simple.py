"""
Simplified data collection module that creates basic training data
without complex spatial interpolation or UCI dataset processing.
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from pathlib import Path

from config import DataConfig, DatabaseConfig

logger = logging.getLogger(__name__)

class SimpleDataCollector:
    """Creates a simple, robust dataset for wildfire prediction"""
    
    def __init__(self):
        self.db_path = DatabaseConfig.DATABASE_PATH
        
    def init_database(self):
        """Initialize simple database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple weather data table
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
        
        # Simple fire records table
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
        
        # Simple satellite data table
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
                UNIQUE(date, location_id)
            )
        ''')
        
        # Training data table (combined features for ML)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                precipitation REAL,
                pressure REAL,
                ndvi REAL,
                evi REAL,
                fire_occurred INTEGER DEFAULT 0,
                fire_risk_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Simple database initialized successfully")
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Create a comprehensive training dataset with realistic patterns"""
        logger.info("Creating comprehensive training dataset...")
        
        # Date range: 3 years of data
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # California coordinates (focus on fire-prone areas)
        locations = [
            {'id': 'NORCAL_01', 'lat': 39.5, 'lon': -121.5, 'name': 'Northern California'},
            {'id': 'NORCAL_02', 'lat': 40.2, 'lon': -122.8, 'name': 'Shasta County'},
            {'id': 'CENTRAL_01', 'lat': 37.5, 'lon': -120.3, 'name': 'Sierra Nevada'},
            {'id': 'CENTRAL_02', 'lat': 36.8, 'lon': -119.4, 'name': 'Central Valley'},
            {'id': 'SOCAL_01', 'lat': 34.2, 'lon': -118.5, 'name': 'Los Angeles County'},
            {'id': 'SOCAL_02', 'lat': 33.8, 'lon': -117.8, 'name': 'Orange County'},
            {'id': 'DESERT_01', 'lat': 34.1, 'lon': -116.2, 'name': 'Mojave Desert'},
            {'id': 'COAST_01', 'lat': 36.6, 'lon': -121.9, 'name': 'Central Coast'},
        ]
        
        training_data = []
        fire_records = []
        fire_id = 0
        
        for date in date_range:
            for location in locations:
                # Generate realistic weather based on location and season
                weather = self._generate_weather(location, date)
                
                # Generate vegetation indices based on season and location
                vegetation = self._generate_vegetation(location, date)
                
                # Determine fire risk and occurrence
                fire_risk = self._calculate_fire_risk(weather, vegetation, date)
                fire_occurred = 1 if np.random.random() < fire_risk else 0
                
                # Create training record
                record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'location_id': location['id'],
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'temperature': weather['temperature'],
                    'humidity': weather['humidity'],
                    'wind_speed': weather['wind_speed'],
                    'precipitation': weather['precipitation'],
                    'pressure': weather['pressure'],
                    'ndvi': vegetation['ndvi'],
                    'evi': vegetation['evi'],
                    'ndmi': vegetation['ndmi'],
                    'nbr': vegetation['nbr'],
                    'fire_occurred': fire_occurred,
                    'fire_risk_score': fire_risk
                }
                training_data.append(record)
                
                # If fire occurred, create fire record
                if fire_occurred:
                    fire_record = {
                        'fire_id': f'FIRE_{fire_id:05d}',
                        'date': date.strftime('%Y-%m-%d'),
                        'latitude': location['lat'] + np.random.normal(0, 0.01),
                        'longitude': location['lon'] + np.random.normal(0, 0.01),
                        'acres_burned': max(0.1, np.random.lognormal(2, 1.5)),
                        'cause': np.random.choice(['Lightning', 'Human', 'Equipment', 'Unknown'], 
                                                p=[0.2, 0.5, 0.2, 0.1]),
                        'confidence': np.random.uniform(0.7, 1.0)
                    }
                    fire_records.append(fire_record)
                    fire_id += 1
        
        # Convert to DataFrames
        training_df = pd.DataFrame(training_data)
        fire_df = pd.DataFrame(fire_records)
        
        logger.info(f"Created training dataset: {len(training_df)} records")
        logger.info(f"Created fire records: {len(fire_df)} records")
        logger.info(f"Fire occurrence rate: {training_df['fire_occurred'].mean():.4f}")
        
        return training_df, fire_df
    
    def _generate_weather(self, location: dict, date: datetime) -> dict:
        """Generate realistic weather for a location and date"""
        
        # Base temperature varies by latitude and season
        lat_factor = (location['lat'] - 36) / 10  # Temperature gradient
        seasonal_temp = 15 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365)
        base_temp = 20 - lat_factor * 5 + seasonal_temp
        
        # Add daily variation and noise
        temperature = base_temp + np.random.normal(0, 3)
        
        # Humidity (inversely related to temperature, varies by coastal proximity)
        coastal_factor = 1 if location['lon'] > -120 else 0.7  # Inland areas drier
        humidity = max(10, min(95, 70 - (temperature - 20) * 1.2 + coastal_factor * 15 + np.random.normal(0, 8)))
        
        # Wind speed (higher in mountain/coastal areas)
        if 'Desert' in location['name'] or 'Coast' in location['name']:
            wind_base = 15
        else:
            wind_base = 10
        wind_speed = max(0, np.random.gamma(2, wind_base/2))
        
        # Precipitation (seasonal, more in winter)
        if date.month in [11, 12, 1, 2, 3]:  # Wet season
            precip_prob = 0.25 if location['lat'] > 37 else 0.15  # More rain in north
        else:  # Dry season
            precip_prob = 0.03
        
        precipitation = np.random.exponential(3) if np.random.random() < precip_prob else 0
        
        # Pressure (varies with elevation approximation)
        base_pressure = 1013 - location['lat'] * 2  # Rough elevation approximation
        pressure = base_pressure + np.random.normal(0, 10)
        
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 1),
            'precipitation': round(precipitation, 2),
            'pressure': round(pressure, 1)
        }
    
    def _generate_vegetation(self, location: dict, date: datetime) -> dict:
        """Generate realistic vegetation indices"""
        
        # Seasonal vegetation cycle
        seasonal_factor = 0.3 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 120) / 365)
        
        # Location-based vegetation (desert vs forest)
        if 'Desert' in location['name']:
            base_ndvi = 0.2
        elif 'Coast' in location['name']:
            base_ndvi = 0.6
        else:
            base_ndvi = 0.5
        
        # Add seasonal variation and noise
        ndvi = max(0, min(1, base_ndvi + seasonal_factor + np.random.normal(0, 0.1)))
        evi = max(0, min(1, ndvi * 0.8 + np.random.normal(0, 0.05)))
        ndmi = max(0, min(1, ndvi * 0.7 + np.random.normal(0, 0.08)))
        nbr = max(0, min(1, ndvi * 0.9 + np.random.normal(0, 0.06)))
        
        return {
            'ndvi': round(ndvi, 3),
            'evi': round(evi, 3),
            'ndmi': round(ndmi, 3),
            'nbr': round(nbr, 3)
        }
    
    def _calculate_fire_risk(self, weather: dict, vegetation: dict, date: datetime) -> float:
        """Calculate fire risk based on weather and vegetation conditions"""
        
        # Base fire risk factors
        temp_risk = max(0, (weather['temperature'] - 25) / 20)  # Higher temp = higher risk
        humidity_risk = max(0, (60 - weather['humidity']) / 50)  # Lower humidity = higher risk
        wind_risk = min(1, weather['wind_speed'] / 30)  # Higher wind = higher risk
        precip_risk = max(0, 1 - weather['precipitation'] / 5)  # Less rain = higher risk
        
        # Vegetation dryness risk (lower NDVI = higher risk)
        veg_risk = max(0, (0.5 - vegetation['ndvi']) / 0.5)
        
        # Seasonal risk (fire season: June-October)
        if date.month in [6, 7, 8, 9, 10]:
            seasonal_risk = 1.0
        elif date.month in [4, 5, 11]:
            seasonal_risk = 0.5
        else:
            seasonal_risk = 0.1
        
        # Combine all risk factors
        combined_risk = (
            temp_risk * 0.25 +
            humidity_risk * 0.25 +
            wind_risk * 0.15 +
            precip_risk * 0.15 +
            veg_risk * 0.20
        ) * seasonal_risk
        
        # Add some random variation
        final_risk = max(0, min(1, combined_risk + np.random.normal(0, 0.1)))
        
        # Scale to realistic fire occurrence rates (1-5% during fire season)
        if seasonal_risk == 1.0:
            return final_risk * 0.05  # Max 5% chance during fire season
        else:
            return final_risk * 0.01  # Max 1% chance outside fire season
    
    def save_to_database(self, training_df: pd.DataFrame, fire_df: pd.DataFrame):
        """Save data to database tables"""
        conn = sqlite3.connect(self.db_path)
        
        # Save training data
        training_df.to_sql('training_data', conn, if_exists='append', index=False)
        logger.info(f"Saved {len(training_df)} training records")
        
        # Save fire records
        fire_df.to_sql(DatabaseConfig.FIRE_TABLE, conn, if_exists='append', index=False)
        logger.info(f"Saved {len(fire_df)} fire records")
        
        # Create separate weather and satellite tables from training data
        weather_cols = ['date', 'location_id', 'latitude', 'longitude', 'temperature', 
                       'humidity', 'wind_speed', 'precipitation', 'pressure']
        weather_df = training_df[weather_cols].copy()
        weather_df.to_sql(DatabaseConfig.WEATHER_TABLE, conn, if_exists='append', index=False)
        logger.info(f"Saved {len(weather_df)} weather records")
        
        satellite_cols = ['date', 'location_id', 'latitude', 'longitude', 'ndvi', 'evi', 'ndmi', 'nbr']
        satellite_df = training_df[satellite_cols].copy()
        satellite_df.to_sql(DatabaseConfig.SATELLITE_TABLE, conn, if_exists='append', index=False)
        logger.info(f"Saved {len(satellite_df)} satellite records")
        
        conn.close()

def main():
    """Main function for simple data collection"""
    logger.info("Starting simple data collection pipeline")
    
    collector = SimpleDataCollector()
    
    # Initialize database
    collector.init_database()
    
    # Create training dataset
    training_df, fire_df = collector.create_training_dataset()
    
    # Save to database
    collector.save_to_database(training_df, fire_df)
    
    logger.info("Simple data collection completed successfully")
    
    # Print summary
    logger.info("="*50)
    logger.info("DATASET SUMMARY")
    logger.info("="*50)
    logger.info(f"Training records: {len(training_df):,}")
    logger.info(f"Fire incidents: {len(fire_df):,}")
    logger.info(f"Fire occurrence rate: {training_df['fire_occurred'].mean():.4f}")
    logger.info(f"Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    logger.info(f"Locations: {training_df['location_id'].nunique()}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
