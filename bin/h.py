# Enhanced web API for wildfire predictions (app.py)
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging

app = Flask(__name__)

# Load all models once at startup
models = {}
model_files = ['gradient_boosting.pkl', 'random_forest.pkl',
               'neural_network.pkl', 'logistic_regression.pkl']

for model_file in model_files:
    model_name = model_file.replace('.pkl', '')
    with open(f'models/{model_file}', 'rb') as f:
        models[model_name] = pickle.load(f)

@app.route('/')
def dashboard():
    """Main dashboard with interactive interface"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_wildfire_risk():
    """Enhanced API endpoint with ensemble and validation"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['temperature', 'humidity', 'wind_speed',
                           'latitude', 'longitude', 'ndvi', 'precipitation']

        # Prepare features (30 engineered features)
        X = prepare_input_features(data)

        # Make predictions with all models
        predictions = {}
        for model_name, model in models.items():
            pred_proba = model.predict_proba(X)[0][1]
            predictions[model_name] = float(pred_proba)

        # Apply physics-based validation
        validated_predictions = validate_predictions(predictions, data)

        # Calculate weighted ensemble
        ensemble_score = calculate_weighted_ensemble(validated_predictions, data)

        # Determine risk level and color
        risk_level = get_risk_level(ensemble_score)
        risk_color = get_risk_color(risk_level)

        response = {
            "risk_score": round(ensemble_score, 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "predictions": validated_predictions,
            "input_data": data,
            "models_used": len(validated_predictions),
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/health')
def health_check():
    """System health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "model_names": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)

# To test: curl -X POST -H "Content-Type: application/json"
#          -d '{"temperature":40,"humidity":15,"wind_speed":20,"latitude":33.65,
#               "longitude":-116.5,"ndvi":0.2,"precipitation":0}'
#          http://localhost:5050/api/predict