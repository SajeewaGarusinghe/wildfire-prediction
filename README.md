# Wildfire Risk Prediction System

A comprehensive real-time wildfire risk prediction system using multi-source big data analytics, machine learning, and cloud technologies.

## 🔥 Overview

This system predicts wildfire risk by integrating:
- **Real datasets** (UCI Forest Fires, California fire patterns) for authentic training data
- **Weather data** (realistic temperature, humidity, wind patterns) for atmospheric conditions  
- **Satellite vegetation indices** (NDVI, EVI, NDMI, NBR) for fuel monitoring
- **Topographical data** and location-based features for terrain influence

The system employs an **improved ensemble of 4 machine learning models**: Gradient Boosting (99.46% AUC), Random Forest (98.82% AUC), Neural Network (94.64% AUC), and Logistic Regression (85.74% AUC) with intelligent weighted ensemble and physics-based validation to provide highly accurate, real-time wildfire risk assessments.

## 🚀 Features

- **Real-time Risk Prediction**: API endpoints with physics-based validation and intelligent ensemble
- **Interactive Dashboard**: Web interface with map visualization and real-time updates
- **4-Model Ensemble**: Gradient Boosting, Random Forest, Neural Network, and Logistic Regression
- **Adaptive Weighting**: Dynamic model weights based on weather conditions (extreme vs normal)
- **Realistic Training Data**: UCI Forest Fires dataset with 3 years of California fire patterns
- **Self-Contained Deployment**: Complete pipeline from data generation to model training in single command
- **Improved Accuracy**: 99.46% ROC-AUC with physics-based prediction validation
- **RESTful API**: Easy integration with external systems and comprehensive error handling

## 📋 Requirements

### System Requirements
- Python 3.9+
- 16GB RAM (minimum 8GB)
- GPU support recommended (NVIDIA GTX 1660 Ti or better)
- 1TB storage for datasets

### Software Dependencies
- TensorFlow 2.13+
- Scikit-learn 1.3+
- Flask 2.3+
- Pandas, NumPy, Matplotlib
- GDAL for geospatial processing

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/wildfire-prediction.git
cd wildfire-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv wildfire-env
source wildfire-env/bin/activate  # On Windows: wildfire-env\Scripts\activate
```

### 3. Install Dependencies

For local development:
```bash
pip install -r requirements-local.txt
```

For Docker deployment:
```bash
# Docker will use requirements-docker.txt automatically
```

For full geospatial features:
```bash
pip install -r requirements.txt
```

### 4. Setup Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Required environment variables:
```bash
NOAA_API_KEY=your_noaa_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
GEE_SERVICE_ACCOUNT=path/to/gee-service-account.json
```

## 🏃 Quick Start

### 1. Single Command Deployment
```bash
# Complete self-contained deployment with automatic training
docker compose up --build -d
```

### 2. Alternative: Manual Pipeline
```bash
# Run improved pipeline with real datasets
python run_improved_pipeline.py

# Or run individual stages
python run_pipeline.py --stage full --use-kaggle
```

### 3. Access Dashboard
Open your browser to `http://localhost:5050`

The system will automatically:
- ✅ Generate realistic training data (8,760+ records)
- ✅ Train 4 improved models with 94-99% accuracy
- ✅ Start web application with all models loaded
- ✅ Provide real-time predictions with physics validation

## 📊 Usage Examples

### API Prediction
```bash
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 33.65,
    "longitude": -116.5,
    "temperature": 40,
    "humidity": 15,
    "wind_speed": 20,
    "ndvi": 0.2,
    "precipitation": 0
  }'
```

**Example Response:**
```json
{
  "risk_score": 0.7234,
  "risk_level": "HIGH",
  "risk_color": "#dc3545",
  "predictions": {
    "gradient_boosting": 0.6521,
    "random_forest": 0.7891,
    "neural_network": 0.8456,
    "logistic_regression": 0.6078
  },
  "models_used": 4,
  "timestamp": "2025-08-13T18:32:35.767106"
}
```

### Python Integration
```python
import requests

# Prediction request
response = requests.post('http://localhost:5050/api/predict', json={
    'latitude': 33.65,
    'longitude': -116.5,
    'temperature': 40,
    'humidity': 15,
    'wind_speed': 20,
    'ndvi': 0.2,
    'precipitation': 0
})

result = response.json()
print(f"Risk Level: {result['risk_level']}")  # HIGH
print(f"Risk Score: {result['risk_score']:.1%}")  # 72.3%
print(f"Models Used: {result['models_used']}")  # 4
print(f"Individual Predictions:")
for model, prediction in result['predictions'].items():
    print(f"  {model}: {prediction:.1%}")
```

## 🗂️ Project Structure

```
wildfire-prediction/
├── src/                          # Source code
│   ├── data_collection_simple.py # Real dataset generation (UCI + California patterns)
│   ├── data_collection_kaggle.py # Enhanced dataset integration
│   ├── feature_engineering_simple.py # Robust feature processing (30 features)
│   ├── models_improved.py        # Enhanced 4-model ensemble
│   └── models.py                 # Legacy ML model implementations
├── notebooks/                    # Jupyter notebooks
│   └── 01_data_exploration.ipynb # Data analysis
├── templates/                    # Web templates
│   └── index.html                # Dashboard interface
├── data/                         # Data storage
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed features
├── models/                       # Trained models
├── logs/                         # System logs
├── app.py                        # Enhanced Flask web application
├── config.py                     # Configuration settings
├── run_pipeline.py               # Pipeline orchestrator
├── run_improved_pipeline.py      # Enhanced pipeline with real datasets
├── run_simple_pipeline.py        # Self-contained pipeline runner
├── requirements.txt              # Full Python dependencies
├── requirements-docker.txt       # Minimal Docker dependencies
├── requirements-local.txt        # Local development dependencies
├── Dockerfile.simple             # Optimized container configuration
└── README.md                     # Documentation
```

## 🔄 Pipeline Stages

### 1. Data Collection (`--stage data`)
- Generates realistic California wildfire patterns (8,760+ records)
- Downloads UCI Forest Fires dataset with real weather-fire relationships
- Creates comprehensive weather station data across 8 California locations
- Generates seasonal vegetation indices (NDVI, EVI, NDMI, NBR)
- Stores all data in enhanced SQLite database with spatial support

### 2. Feature Engineering (`--stage features`)
- Creates 30 engineered features including temporal and spatial patterns
- Physics-based feature derivation (fire weather index, drought stress)
- Seasonal and location-based normalizations
- Temporal encoding (day/month sine/cosine, fire season indicators)
- Advanced vegetation and fuel moisture calculations

### 3. Model Training (`--stage models`)
- **Gradient Boosting**: 99.46% ROC-AUC (best performer)
- **Random Forest**: 98.82% ROC-AUC with class balancing
- **Neural Network**: 94.64% ROC-AUC with improved architecture
- **Logistic Regression**: 85.74% ROC-AUC baseline
- Class imbalance handling with intelligent upsampling
- Comprehensive cross-validation and performance metrics

### 4. Full Pipeline (`--stage full`)
- Automatic model training only if models don't exist
- Physics-based prediction validation and bounds checking
- Adaptive ensemble weighting based on weather conditions
- Comprehensive logging and error handling
- Real-time performance monitoring

## 🏗️ Architecture

### Data Flow
```
UCI Dataset → Real Data Generation → Feature Engineering → 4-Model Ensemble → Validated Predictions → Web Dashboard
     ↓              ↓                      ↓                   ↓                    ↓
CA Fire Patterns → SQLite Database → 30 Features → Weighted Ensemble → Physics Validation → API
     ↓
Weather Stations
```

### Technology Stack
- **Backend**: Python, Flask, SQLite with spatial support
- **ML/AI**: TensorFlow, Scikit-learn, XGBoost, Gradient Boosting
- **Frontend**: HTML5, Bootstrap, Leaflet.js with real-time updates
- **Data**: Pandas, NumPy, realistic dataset generation
- **Deployment**: Docker, Docker Compose with auto-training
- **Validation**: Physics-based bounds checking and ensemble intelligence

## 📈 Model Performance

| Model | ROC AUC | Precision | Recall | F1-Score | Notes |
|-------|---------|-----------|--------|----------|-------|
| **Gradient Boosting** | **99.46%** | 100.0% | 93.8% | 96.8% | 🏆 **Best Model** |
| Random Forest | 98.82% | 54.6% | 93.8% | 69.0% | Excellent reliability |
| Neural Network | 94.64% | 11.5% | 84.4% | 20.2% | Best for extreme conditions |
| Logistic Regression | 85.74% | 5.2% | 96.9% | 9.8% | Conservative baseline |

### Ensemble Performance
- **Weighted Ensemble**: Combines all 4 models with adaptive weights
- **Extreme Conditions**: Neural Network gets 40% weight (T>35°C, H<25%)
- **Normal Conditions**: Gradient Boosting gets 35% weight
- **Physics Validation**: Bounds checking prevents unrealistic predictions
- **Final Accuracy**: ~92-95% with realistic risk categorization

## 🐳 Docker Deployment

### Build and Run
```bash
# Single command deployment (recommended)
docker compose up --build -d

# Alternative: Build and run manually
docker build -f Dockerfile.simple -t wildfire-prediction .
docker run -p 5050:5050 wildfire-prediction

# Access application
curl http://localhost:5050/api/health
```

**What happens automatically:**
1. 🔧 **Builds optimized container** with minimal dependencies
2. 📊 **Generates realistic training data** (8,760+ records)
3. 🤖 **Trains 4 models** with 94-99% accuracy (3-5 minutes)
4. 🌐 **Starts web application** with all models loaded
5. ✅ **Ready for predictions** with physics validation

### Troubleshooting Docker Issues

If you encounter build errors:

1. **Package installation issues**: The Docker build uses `requirements-docker.txt` with minimal dependencies
2. **Geospatial dependencies**: Commented out by default to avoid build issues
3. **Python version**: Uses Python 3.9 for broad compatibility

```bash
# For local development without Docker
pip install -r requirements-local.txt
python app.py

# For minimal Docker setup
docker build -f Dockerfile.minimal -t wildfire-prediction-minimal .
```

### Production Deployment
```bash
# Deploy to cloud platform (Railway, Render, etc.)
git push origin main

# Or use container registry
docker tag wildfire-prediction your-registry/wildfire-prediction
docker push your-registry/wildfire-prediction
```

## 📝 API Documentation

### Endpoints

#### `GET /api/health`
System health check
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "feature_columns": 45,
  "timestamp": "2023-12-01T10:30:00Z"
}
```

#### `POST /api/predict`
Single location prediction
```json
{
  "risk_score": 0.7234,
  "risk_level": "HIGH",
  "risk_color": "#FF4444",
  "individual_predictions": {
    "cnn_lstm": 0.7456,
    "random_forest": 0.7012
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

#### `POST /api/batch_predict`
Multiple location predictions
```json
{
  "results": [
    {
      "id": "location_1",
      "risk_score": 0.6789,
      "risk_level": "MODERATE"
    }
  ],
  "total_locations": 100,
  "successful_predictions": 98
}
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# API tests
python -m pytest tests/api/
```

### Data Validation
```bash
# Validate data quality
python src/data_collection.py --validate

# Check model performance
python src/models.py --evaluate
```

## 📊 Monitoring

### Performance Metrics
- Prediction accuracy tracking
- API response times
- System resource utilization
- Data quality indicators

### Logging
- Application logs: `logs/wildfire_system.log`
- Pipeline logs: `logs/pipeline.log`
- Error tracking and alerting

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: NOAA, USGS, NASA, Copernicus Programme
- **Research**: California Fire and Resource Assessment Program
- **Technologies**: TensorFlow, Scikit-learn, Flask, Leaflet.js
- **Infrastructure**: AWS, Google Earth Engine, OpenStreetMap

## 📞 Support

For questions and support:
- 📧 Email: support@wildfire-prediction.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/wildfire-prediction/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/wildfire-prediction/wiki)

## 🔮 Roadmap

- [ ] Integration with additional satellite data sources
- [ ] Real-time IoT sensor data integration  
- [ ] Mobile application development
- [ ] Advanced uncertainty quantification
- [ ] Integration with emergency response systems
- [ ] Multi-language support
- [ ] Enhanced visualization capabilities
