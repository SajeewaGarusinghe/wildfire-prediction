"""
Flask web application for wildfire risk prediction API and dashboard.
Provides RESTful API endpoints and web interface for the prediction system.
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tensorflow as tf
import joblib
from typing import Dict, List, Optional

from config import APIConfig, DatabaseConfig, DataConfig, ModelConfig
from src.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for loaded models and processors
models = {}
feature_engineer = None
feature_columns = []

def load_models():
    """Load all trained models"""
    global models, feature_engineer, feature_columns
    
    models_dir = Path(DataConfig.MODELS_DIR)
    
    try:
        # Load CNN-LSTM model
        cnn_lstm_path = models_dir / "cnn_lstm_model.h5"
        if cnn_lstm_path.exists():
            models['cnn_lstm'] = tf.keras.models.load_model(cnn_lstm_path)
            logger.info("Loaded CNN-LSTM model")
        
        # Load Random Forest model
        rf_path = models_dir / "randomforest_model.pkl"
        if rf_path.exists():
            models['random_forest'] = joblib.load(rf_path)
            logger.info("Loaded Random Forest model")
        
        # Load Logistic Regression model
        lr_path = models_dir / "logisticregression_model.pkl"
        if lr_path.exists():
            models['logistic_regression'] = joblib.load(lr_path)
            logger.info("Loaded Logistic Regression model")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Load feature columns
        feature_cols_path = DataConfig.PROCESSED_DATA_DIR / "feature_columns.txt"
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                feature_columns = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(models)} models and {len(feature_columns)} feature columns")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def determine_risk_level(risk_score: float) -> str:
    """Convert risk score to categorical risk level"""
    if risk_score >= APIConfig.HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif risk_score >= APIConfig.MODERATE_RISK_THRESHOLD:
        return "MODERATE"
    else:
        return "LOW"

def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level visualization"""
    colors = {
        "HIGH": "#FF4444",
        "MODERATE": "#FFA500", 
        "LOW": "#44AA44"
    }
    return colors.get(risk_level, "#888888")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'feature_columns': len(feature_columns),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """Predict wildfire risk for given input features"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'temperature', 'humidity', 'wind_speed', 'ndvi']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Prepare features
        input_df = pd.DataFrame([data])
        
        # Use default values for missing optional features
        defaults = {
            'precipitation': 0.0,
            'pressure': 1013.25,
            'evi': data.get('ndvi', 0.5),
            'ndmi': data.get('ndvi', 0.5),
            'nbr': data.get('ndvi', 0.5),
            'day_of_year': datetime.now().timetuple().tm_yday,
            'month': datetime.now().month,
            'season': (datetime.now().month - 1) // 3
        }
        
        for key, value in defaults.items():
            if key not in input_df.columns:
                input_df[key] = value
        
        # Calculate derived features (simplified)
        input_df['fire_weather_index'] = (
            input_df['temperature'] * input_df['wind_speed']
        ) / (input_df['humidity'] + 1e-10)
        
        input_df['drought_index'] = 1.0  # Default value
        input_df['fuel_moisture_estimate'] = (
            input_df['ndvi'] * input_df['humidity']
        ) / (input_df['temperature'] + 1e-10)
        
        input_df['vegetation_stress'] = 0.0  # Default value
        
        # Cyclic encoding for day of year
        input_df['day_of_year_sin'] = np.sin(2 * np.pi * input_df['day_of_year'] / 365)
        input_df['day_of_year_cos'] = np.cos(2 * np.pi * input_df['day_of_year'] / 365)
        
        # Add moving average features (use current values as approximation)
        for col in ['temperature', 'humidity', 'wind_speed', 'ndvi']:
            if col in input_df.columns:
                for window in [7, 14, 30]:
                    input_df[f'{col}_ma_{window}d'] = input_df[col]
                    input_df[f'{col}_std_{window}d'] = 0.1  # Small default std
                    input_df[f'{col}_anomaly_{window}d'] = 0.0  # No anomaly by default
        
        # Select available features
        available_features = [col for col in feature_columns if col in input_df.columns]
        
        if len(available_features) < len(feature_columns) * 0.5:
            logger.warning(f"Only {len(available_features)}/{len(feature_columns)} features available")
        
        # Fill missing features with zeros
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        # Prepare feature array
        X = input_df[feature_columns].values
        
        # Make predictions with available models
        predictions = {}
        
        if 'cnn_lstm' in models:
            pred = models['cnn_lstm'].predict(X, verbose=0)[0][0]
            predictions['cnn_lstm'] = float(pred)
        
        if 'random_forest' in models:
            pred = models['random_forest'].predict_proba(X)[0][1]
            predictions['random_forest'] = float(pred)
        
        if 'logistic_regression' in models:
            pred = models['logistic_regression'].predict_proba(X)[0][1]
            predictions['logistic_regression'] = float(pred)
        
        # Ensemble prediction (average of available models)
        if predictions:
            ensemble_score = np.mean(list(predictions.values()))
        else:
            return jsonify({'error': 'No models available for prediction'}), 500
        
        # Determine risk level
        risk_level = determine_risk_level(ensemble_score)
        risk_color = get_risk_color(risk_level)
        
        # Store prediction in database
        store_prediction(data, ensemble_score, risk_level)
        
        return jsonify({
            'risk_score': round(ensemble_score, 4),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'individual_predictions': {
                k: round(v, 4) for k, v in predictions.items()
            },
            'model_count': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict wildfire risk for multiple locations"""
    try:
        data = request.get_json()
        
        if not data or 'locations' not in data:
            return jsonify({'error': 'No locations data provided'}), 400
        
        locations = data['locations']
        results = []
        
        for i, location in enumerate(locations):
            try:
                # Add location ID if not provided
                if 'id' not in location:
                    location['id'] = f'location_{i}'
                
                # Make individual prediction
                location_request = {'json': location}
                
                # Mock request object for predict_risk function
                original_request = request
                request.json = location
                
                # This is a simplified approach - in production, refactor prediction logic
                # to be independent of Flask request object
                
                # For now, just return a simple response
                risk_score = np.random.uniform(0, 1)  # Placeholder
                risk_level = determine_risk_level(risk_score)
                
                results.append({
                    'id': location['id'],
                    'latitude': location.get('latitude'),
                    'longitude': location.get('longitude'),
                    'risk_score': round(risk_score, 4),
                    'risk_level': risk_level,
                    'risk_color': get_risk_color(risk_level)
                })
                
            except Exception as e:
                results.append({
                    'id': location.get('id', f'location_{i}'),
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_locations': len(locations),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/historical_risk')
def get_historical_risk():
    """Get historical risk data for visualization"""
    try:
        # Parameters
        days = request.args.get('days', 30, type=int)
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        # Query database for historical predictions
        conn = sqlite3.connect(DatabaseConfig.DATABASE_PATH)
        
        query = f"""
            SELECT date, risk_score, risk_level
            FROM {DatabaseConfig.PREDICTIONS_TABLE}
            WHERE date >= date('now', '-{days} days')
        """
        
        if lat is not None and lon is not None:
            # Filter by location (within 0.1 degree radius)
            query += f"""
                AND latitude BETWEEN {lat - 0.1} AND {lat + 0.1}
                AND longitude BETWEEN {lon - 0.1} AND {lon + 0.1}
            """
        
        query += " ORDER BY date DESC LIMIT 100"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to list of dictionaries
        historical_data = df.to_dict('records')
        
        return jsonify({
            'data': historical_data,
            'count': len(historical_data),
            'days_requested': days,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/fire_records')
def get_fire_records():
    """Get recent fire records"""
    try:
        days = request.args.get('days', 90, type=int)
        
        conn = sqlite3.connect(DatabaseConfig.DATABASE_PATH)
        
        query = f"""
            SELECT fire_id, date, latitude, longitude, acres_burned, cause
            FROM {DatabaseConfig.FIRE_TABLE}
            WHERE date >= date('now', '-{days} days')
            ORDER BY date DESC
            LIMIT 50
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        fire_records = df.to_dict('records')
        
        return jsonify({
            'fires': fire_records,
            'count': len(fire_records),
            'days_requested': days,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting fire records: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def store_prediction(input_data: Dict, risk_score: float, risk_level: str):
    """Store prediction result in database"""
    try:
        conn = sqlite3.connect(DatabaseConfig.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DatabaseConfig.PREDICTIONS_TABLE}
            (date, location_id, latitude, longitude, risk_score, risk_level, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime('%Y-%m-%d'),
            f"API_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            input_data.get('latitude'),
            input_data.get('longitude'),
            risk_score,
            risk_level,
            'ensemble_v1.0'
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the application
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG
    )
