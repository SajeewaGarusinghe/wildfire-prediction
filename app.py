"""
Updated Flask web application for wildfire risk prediction.
Enhanced to work with simple models.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from datetime import datetime
import sqlite3

# Optional imports with fallback
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
try:
    from src.feature_engineering_simple import SimpleFeatureEngineer
    SIMPLE_FE_AVAILABLE = True
except ImportError:
    SIMPLE_FE_AVAILABLE = False

try:
    from src.feature_engineering import FeatureEngineer
    COMPLEX_FE_AVAILABLE = True
except ImportError:
    COMPLEX_FE_AVAILABLE = False

from config import DataConfig, APIConfig, DatabaseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for loaded models and processors
models = {}
feature_engineer = None
feature_columns = []
scaler = None

def load_models():
    """Load all trained models (simple and complex versions)"""
    global models, feature_engineer, feature_columns, scaler
    
    try:
        models_dir = Path(DataConfig.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Looking for models in: {models_dir}")
        
        # Try to load simple models first (new format)
        simple_models_loaded = False
        
        # Load Simple Random Forest model
        rf_simple_path = models_dir / "random_forest.pkl"
        if rf_simple_path.exists():
            models['random_forest'] = joblib.load(rf_simple_path)
            logger.info("✅ Loaded Simple Random Forest model")
            simple_models_loaded = True
        
        # Load Simple Logistic Regression model  
        lr_simple_path = models_dir / "logistic_regression.pkl"
        if lr_simple_path.exists():
            models['logistic_regression'] = joblib.load(lr_simple_path)
            logger.info("✅ Loaded Simple Logistic Regression model")
            simple_models_loaded = True
        
        # Load Gradient Boosting model
        gb_simple_path = models_dir / "gradient_boosting.pkl"
        if gb_simple_path.exists():
            models['gradient_boosting'] = joblib.load(gb_simple_path)
            logger.info("✅ Loaded Gradient Boosting model")
            simple_models_loaded = True
        
        # Load Simple Neural Network model
        if TF_AVAILABLE:
            nn_simple_path = models_dir / "neural_network.h5"
            if nn_simple_path.exists():
                try:
                    models['neural_network'] = tf.keras.models.load_model(nn_simple_path)
                    logger.info("✅ Loaded Simple Neural Network model")
                    simple_models_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load neural network: {e}")
        
        # Load Simple Feature Scaler
        scaler_path = models_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("✅ Loaded feature scaler")
        
        # Load Simple Feature Columns
        feature_columns_path = models_dir / "feature_columns.pkl"
        if feature_columns_path.exists():
            feature_columns = joblib.load(feature_columns_path)
            logger.info(f"✅ Loaded {len(feature_columns)} feature columns")
        else:
            # Default feature columns for simple models
            feature_columns = [
                'temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure',
                'ndvi', 'evi', 'ndmi', 'nbr',
                'day_of_year', 'month', 'season', 'fire_season',
                'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'temp_humidity_ratio', 'wind_temp_index', 'dryness_index', 'fire_weather_index',
                'vegetation_stress', 'fuel_moisture', 'vegetation_health',
                'drought_stress', 'ignition_potential',
                'lat_normalized', 'lon_normalized', 'distance_from_coast', 'elevation_proxy'
            ]
            logger.warning("Using default simple model feature columns")
        
        # Initialize feature engineer
        if SIMPLE_FE_AVAILABLE and simple_models_loaded:
            feature_engineer = SimpleFeatureEngineer()
            logger.info("✅ Initialized Simple Feature Engineer")
        elif COMPLEX_FE_AVAILABLE:
            feature_engineer = FeatureEngineer()
            logger.info("✅ Initialized Complex Feature Engineer")
        else:
            feature_engineer = None
            logger.warning("⚠️ No Feature Engineer available")
        
        # If no simple models, try legacy models
        if not simple_models_loaded:
            logger.info("No simple models found, trying legacy models...")
            
            # Load CNN-LSTM model (legacy)
            cnn_lstm_path = models_dir / "cnn_lstm_model.h5"
            if cnn_lstm_path.exists() and TF_AVAILABLE:
                models['cnn_lstm'] = tf.keras.models.load_model(cnn_lstm_path)
                logger.info("✅ Loaded Legacy CNN-LSTM model")
            
            # Load Random Forest model (legacy)
            rf_path = models_dir / "randomforest_model.pkl"
            if rf_path.exists():
                models['random_forest'] = joblib.load(rf_path)
                logger.info("✅ Loaded Legacy Random Forest model")
            
            # Load Logistic Regression model (legacy)
            lr_path = models_dir / "logisticregression_model.pkl"
            if lr_path.exists():
                models['logistic_regression'] = joblib.load(lr_path)
                logger.info("✅ Loaded Legacy Logistic Regression model")
        
        if len(models) == 0:
            logger.warning("⚠️ No trained models found. The system will use heuristic predictions.")
        else:
            logger.info(f"🎉 Successfully loaded {len(models)} models")
            logger.info(f"📊 Models available: {list(models.keys())}")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        models = {}
        feature_engineer = None
        scaler = None
        feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'precipitation',
            'ndvi', 'evi', 'ndmi', 'nbr'
        ]

def prepare_input_features(input_data: dict) -> np.ndarray:
    """Prepare input features for prediction using simple feature engineering"""
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Add derived features similar to training - use current date for seasonal features
        current_date = datetime.now()
        df['date'] = pd.to_datetime(current_date)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })
        df['fire_season'] = df['month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10] else 0)
        
        logger.info(f"Current date for prediction: {current_date.strftime('%Y-%m-%d')}, Month: {df['month'].iloc[0]}, Fire season: {df['fire_season'].iloc[0]}")
        
        # Cyclic encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Derived features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['wind_temp_index'] = df['wind_speed'] * df['temperature'] / 100
        df['dryness_index'] = (100 - df['humidity']) * df['temperature'] / 100
        df['fire_weather_index'] = (
            df['temperature'] * 0.3 + 
            (100 - df['humidity']) * 0.3 + 
            df['wind_speed'] * 0.2 + 
            (1 / (df.get('precipitation', 0) + 0.1)) * 0.2
        )
        
        # Vegetation features (use defaults if not provided)
        if 'ndvi' not in df.columns:
            df['ndvi'] = 0.5
        if 'evi' not in df.columns:
            df['evi'] = 0.4
        if 'ndmi' not in df.columns:
            df['ndmi'] = 0.3
        if 'nbr' not in df.columns:
            df['nbr'] = 0.6
            
        df['vegetation_stress'] = (1 - df['ndvi']) * (1 - df['ndmi'])
        df['fuel_moisture'] = df['ndmi'] * df['humidity'] / 100
        df['vegetation_health'] = (df['ndvi'] + df['evi']) / 2
        
        # Risk indicators
        df['drought_stress'] = (
            (100 - df['humidity']) / 100 * 0.4 +
            (1 - df['ndmi']) * 0.3 +
            (1 / (df.get('precipitation', 0) + 0.1)) * 0.3
        )
        df['ignition_potential'] = (
            df['temperature'] / 40 * 0.3 +
            df['wind_speed'] / 30 * 0.3 +
            (100 - df['humidity']) / 100 * 0.4
        )
        
        # Location features (use defaults)
        df['lat_normalized'] = 0.5
        df['lon_normalized'] = 0.5
        df['distance_from_coast'] = 2.0
        df['elevation_proxy'] = 1000.0
        
        # Add missing pressure if not provided
        if 'pressure' not in df.columns:
            df['pressure'] = 1013.0
        
        # Select features that exist in the model
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0.0
            logger.info(f"Added {len(missing_features)} missing features with default values")
        
        # Now use all required features
        X = df[feature_columns].values
        logger.info(f"Using {len(feature_columns)} features for prediction (shape: {X.shape})")
        logger.info(f"Feature sample values: {df[feature_columns[:5]].iloc[0].to_dict()}")
        
        # Scale features if scaler is available
        if scaler is not None:
            X = scaler.transform(X)
            logger.info(f"Features scaled using saved scaler")
        
        return X
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        # Return basic features as fallback
        basic_features = [
            input_data.get('temperature', 25),
            input_data.get('humidity', 50),
            input_data.get('wind_speed', 10),
            input_data.get('precipitation', 0),
            input_data.get('pressure', 1013),
            input_data.get('ndvi', 0.5),
            input_data.get('evi', 0.4),
            input_data.get('ndmi', 0.3),
            input_data.get('nbr', 0.6)
        ]
        return np.array(basic_features).reshape(1, -1)

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
        "LOW": "#28a745",      # Green
        "MODERATE": "#ffc107",  # Yellow
        "HIGH": "#dc3545"       # Red
    }
    return colors.get(risk_level, "#6c757d")

def validate_predictions(raw_predictions: dict, input_data: dict) -> dict:
    """Validate model predictions and filter out obviously broken ones"""
    validated = {}
    
    # Calculate expected risk range based on conditions
    temp = input_data.get('temperature', 25)
    humidity = input_data.get('humidity', 50)
    wind = input_data.get('wind_speed', 10)
    precip = input_data.get('precipitation', 0)
    
    # Simple physics-based risk estimate for sanity checking
    temp_risk = max(0, min(1, (temp - 10) / 40))  # 0% at 10°C, 100% at 50°C
    humidity_risk = max(0, min(1, (80 - humidity) / 80))  # 0% at 80%, 100% at 0%
    wind_risk = max(0, min(1, wind / 40))  # 0% at 0mph, 100% at 40mph
    precip_risk = max(0, min(1, (5 - precip) / 5))  # 0% at 5mm+, 100% at 0mm
    
    physics_risk = (temp_risk + humidity_risk + wind_risk + precip_risk) / 4
    
    # Define reasonable bounds
    min_expected = max(0.001, physics_risk * 0.1)  # At least 10% of physics estimate
    max_expected = min(0.95, physics_risk * 3 + 0.1)  # At most 3x physics + baseline
    
    logger.info(f"Physics-based risk estimate: {physics_risk:.3f}, bounds: [{min_expected:.3f}, {max_expected:.3f}]")
    
    for model_name, prediction in raw_predictions.items():
        
        # Check for obviously broken predictions
        if model_name == 'logistic_regression':
            # Logistic regression seems to always predict very high values
            if prediction > 0.9 and physics_risk < 0.5:
                logger.warning(f"Logistic regression prediction {prediction:.3f} seems too high for conditions, capping at {max_expected:.3f}")
                validated[model_name] = min(prediction, max_expected)
            else:
                validated[model_name] = prediction
        
        elif model_name in ['gradient_boosting', 'random_forest']:
            # Tree models seem to predict too low for extreme conditions
            if prediction < 0.05 and physics_risk > 0.7:
                logger.warning(f"{model_name} prediction {prediction:.3f} seems too low for extreme conditions, boosting")
                validated[model_name] = max(prediction, min_expected)
            else:
                validated[model_name] = prediction
        
        else:
            # Neural network and others - apply general bounds
            if prediction < min_expected or prediction > max_expected:
                clamped = max(min_expected, min(prediction, max_expected))
                logger.warning(f"{model_name} prediction {prediction:.3f} outside expected range, clamped to {clamped:.3f}")
                validated[model_name] = clamped
            else:
                validated[model_name] = prediction
    
    return validated

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "features": len(feature_columns),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/models')
def get_models_info():
    """Get information about loaded models"""
    model_info = {}
    for name, model in models.items():
        if hasattr(model, 'get_params'):
            model_info[name] = {
                "type": type(model).__name__,
                "parameters": model.get_params()
            }
        else:
            model_info[name] = {
                "type": type(model).__name__,
                "parameters": "TensorFlow model"
            }
    
    return jsonify({
        "models": model_info,
        "feature_count": len(feature_columns),
        "features": feature_columns[:10],  # First 10 features
        "scaler_loaded": scaler is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_wildfire_risk():
    """Predict wildfire risk based on input conditions"""
    try:
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate required fields
        required_fields = ['temperature', 'humidity', 'wind_speed']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Prepare features
        X = prepare_input_features(data)
        
        # Make predictions with all available models
        raw_predictions = {}
        predictions = {}
        
        for model_name, model in models.items():
            try:
                if model_name == 'neural_network' and TF_AVAILABLE:
                    # TensorFlow model
                    pred_proba = model.predict(X, verbose=0)[0][0]
                    raw_predictions[model_name] = float(pred_proba)
                else:
                    # Scikit-learn model
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)[0][1]  # Probability of fire
                        raw_predictions[model_name] = float(pred_proba)
                    else:
                        pred = model.predict(X)[0]
                        raw_predictions[model_name] = float(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        # Validate and filter predictions
        predictions = validate_predictions(raw_predictions, data)
        logger.info(f"Raw predictions: {raw_predictions}")
        logger.info(f"Validated predictions: {predictions}")
        
        # Calculate weighted ensemble prediction based on model performance
        if predictions:
            # Adjust weights based on which models are working properly
            # Neural network seems most reliable for extreme conditions
            
            temp = data.get('temperature', 25)
            humidity = data.get('humidity', 50)
            
            if temp > 35 or humidity < 25:  # Extreme conditions
                # Give more weight to neural network during extreme conditions
                model_weights = {
                    'neural_network': 0.4,       # Most reliable for extremes
                    'gradient_boosting': 0.25,   # Good but conservative
                    'random_forest': 0.25,       # Good but conservative  
                    'logistic_regression': 0.1   # Tends to overpredict
                }
            else:  # Normal conditions
                # Standard weights for normal conditions
                model_weights = {
                    'gradient_boosting': 0.35,   # Best overall (99.46% AUC)
                    'random_forest': 0.30,       # Second best (98.82% AUC)  
                    'neural_network': 0.25,      # Good performance (94.64% AUC)
                    'logistic_regression': 0.10  # Baseline (85.74% AUC)
                }
            
            # Calculate weighted average
            weighted_sum = 0
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                weight = model_weights.get(model_name, 0.2)  # Default weight
                weighted_sum += prediction * weight
                total_weight += weight
            
            ensemble_score = weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
            
            # Apply extreme condition boost for very high-risk scenarios
            temp = data.get('temperature', 25)
            humidity = data.get('humidity', 50)
            wind = data.get('wind_speed', 10)
            
            # Extreme fire weather conditions boost
            if temp > 35 and humidity < 20 and wind > 15:
                extreme_boost = min(0.2, (temp - 35) / 50 + (20 - humidity) / 100 + (wind - 15) / 100)
                ensemble_score = min(0.95, ensemble_score + extreme_boost)
                logger.info(f"Applied extreme conditions boost: +{extreme_boost:.3f}")
        else:
            logger.warning("No models available, using heuristic prediction")
            # Simple fire risk heuristic based on weather conditions
            temp_risk = max(0, (data['temperature'] - 25) / 20)
            humidity_risk = max(0, (60 - data['humidity']) / 50)
            wind_risk = min(1, data['wind_speed'] / 30)
            
            ensemble_score = (temp_risk + humidity_risk + wind_risk) / 3 * 0.1
            predictions['heuristic'] = float(ensemble_score)
        
        # Determine risk level and color
        risk_level = determine_risk_level(ensemble_score)
        risk_color = get_risk_color(risk_level)
        
        # Prepare response
        response = {
            "risk_score": round(ensemble_score, 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "predictions": predictions,
            "input_data": data,
            "models_used": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Raw predictions: {raw_predictions}")
        logger.info(f"Validated predictions: {predictions}")
        logger.info(f"Weighted ensemble: {ensemble_score:.4f} ({risk_level}) using {len(predictions)} models")
        logger.info(f"Input conditions: T={data.get('temperature')}°C, H={data.get('humidity')}%, W={data.get('wind_speed')}mph, P={data.get('precipitation')}mm")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/api/data/summary')
def get_data_summary():
    """Get summary of training data"""
    try:
        conn = sqlite3.connect(DatabaseConfig.DATABASE_PATH)
        
        # Get training data summary
        summary_query = """
        SELECT 
            COUNT(*) as total_records,
            SUM(fire_occurred) as fire_events,
            AVG(temperature) as avg_temp,
            AVG(humidity) as avg_humidity,
            AVG(wind_speed) as avg_wind,
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(DISTINCT location_id) as locations
        FROM training_data
        """
        
        result = pd.read_sql_query(summary_query, conn)
        conn.close()
        
        summary = result.iloc[0].to_dict()
        summary['fire_rate'] = summary['fire_events'] / summary['total_records'] if summary['total_records'] > 0 else 0
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return jsonify({"error": "Failed to get data summary"}), 500

# Load models on startup
load_models()

if __name__ == '__main__':
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG
    )