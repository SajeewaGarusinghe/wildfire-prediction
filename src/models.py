"""
Machine learning models for wildfire risk prediction.
Includes CNN-LSTM ensemble, Random Forest baseline, and model evaluation.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

from config import ModelConfig, DataConfig
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.metrics = {}
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        raise NotImplementedError
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        # For binary classification
        if len(np.unique(y)) == 2:
            return {
                'accuracy': accuracy_score(y, (predictions > 0.5).astype(int)),
                'precision': precision_score(y, (predictions > 0.5).astype(int)),
                'recall': recall_score(y, (predictions > 0.5).astype(int)),
                'f1_score': f1_score(y, (predictions > 0.5).astype(int)),
                'roc_auc': roc_auc_score(y, predictions)
            }
        # For regression
        else:
            return {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': sklearn.metrics.r2_score(y, predictions)
            }

class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for spatial-temporal wildfire prediction"""
    
    def __init__(self, input_shape: Tuple[int, ...], **kwargs):
        super().__init__("CNN_LSTM")
        self.input_shape = input_shape
        self.params = {**ModelConfig.CNN_LSTM_PARAMS, **kwargs}
        self.build_model()
    
    def build_model(self):
        """Build CNN-LSTM architecture"""
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # Reshape for CNN if needed (assume we have spatial structure)
        # For demo, we'll use 1D CNN followed by LSTM
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        
        # CNN layers
        for i, filters in enumerate(self.params['cnn_filters']):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=self.params['cnn_kernel_size'],
                activation='relu',
                padding='same',
                name=f'conv1d_{i}'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # LSTM layers
        x = layers.LSTM(
            units=self.params['lstm_units'],
            return_sequences=False,
            dropout=self.params['dropout_rate'],
            name='lstm'
        )(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Built CNN-LSTM model with {self.model.count_params()} parameters")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model"""
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Return training history
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'params': self.params,
            'metrics': self.metrics
        }
        
        with open(filepath.parent / f"{filepath.stem}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        
        # Load metadata if available
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.metrics = metadata.get('metrics', {})

class RandomForestModel(BaseModel):
    """Random Forest baseline model"""
    
    def __init__(self, task: str = 'classification', **kwargs):
        super().__init__("RandomForest")
        self.task = task
        self.params = {**ModelConfig.RANDOM_FOREST_PARAMS, **kwargs}
        
        if task == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        feature_importance = self.model.feature_importances_
        
        return {'feature_importance': feature_importance}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.task == 'classification':
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'task': self.task,
            'params': self.params,
            'metrics': self.metrics
        }
        
        with open(filepath.parent / f"{filepath.stem}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

class LogisticRegressionModel(BaseModel):
    """Simple logistic regression baseline"""
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression")
        self.params = kwargs
        self.model = LogisticRegression(random_state=ModelConfig.RANDOM_SEED, **kwargs)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

class ModelEvaluator:
    """Evaluate and compare multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, model: BaseModel, name: str = None):
        """Add model for evaluation"""
        name = name or model.model_name
        self.models[name] = model
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all models"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                metrics = model.evaluate(X_test, y_test)
                results[name] = metrics
                
                # Store in model object
                model.metrics = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table of model performance"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                row = {'Model': model_name, **metrics}
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, filepath: Path):
        """Save evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

class ModelTrainer:
    """Main training pipeline for all models"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.feature_columns = None
        
    def prepare_data(self, start_date: str, end_date: str) -> Tuple[np.ndarray, ...]:
        """Prepare training and testing data"""
        logger.info("Preparing features...")
        features_df, self.feature_columns = self.feature_engineer.prepare_features(
            start_date, end_date
        )
        
        # Split by time (2020-2021 train, 2022 val, 2023 test)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        train_mask = features_df['date'] < '2022-01-01'
        val_mask = (features_df['date'] >= '2022-01-01') & (features_df['date'] < '2023-01-01')
        test_mask = features_df['date'] >= '2023-01-01'
        
        train_df = features_df[train_mask]
        val_df = features_df[val_mask]
        test_df = features_df[test_mask]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Prepare features and labels
        X_train = self.feature_engineer.preprocess_features(
            train_df, self.feature_columns, fit_scalers=True
        )
        X_val = self.feature_engineer.preprocess_features(
            val_df, self.feature_columns, fit_scalers=False
        )
        X_test = self.feature_engineer.preprocess_features(
            test_df, self.feature_columns, fit_scalers=False
        )
        
        y_train = train_df['fire_occurred'].values
        y_val = val_df['fire_occurred'].values
        y_test = test_df['fire_occurred'].values
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train all models"""
        results = {}
        
        # CNN-LSTM Model
        logger.info("Training CNN-LSTM model...")
        cnn_lstm = CNNLSTMModel(input_shape=(X_train.shape[1],))
        cnn_lstm_history = cnn_lstm.train(X_train, y_train, X_val, y_val)
        self.models['CNN_LSTM'] = cnn_lstm
        results['CNN_LSTM'] = cnn_lstm_history
        
        # Random Forest
        logger.info("Training Random Forest model...")
        rf_model = RandomForestModel(task='classification')
        rf_history = rf_model.train(X_train, y_train, X_val, y_val)
        self.models['RandomForest'] = rf_model
        results['RandomForest'] = rf_history
        
        # Logistic Regression
        logger.info("Training Logistic Regression model...")
        lr_model = LogisticRegressionModel(max_iter=1000)
        lr_history = lr_model.train(X_train, y_train, X_val, y_val)
        self.models['LogisticRegression'] = lr_model
        results['LogisticRegression'] = lr_history
        
        return results
    
    def save_all_models(self):
        """Save all trained models"""
        models_dir = Path(DataConfig.MODELS_DIR)
        
        for name, model in self.models.items():
            if name == 'CNN_LSTM':
                filepath = models_dir / f"{name.lower()}_model.h5"
            else:
                filepath = models_dir / f"{name.lower()}_model.pkl"
            
            model.save_model(filepath)
            logger.info(f"Saved {name} model to {filepath}")

def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")
    
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        DataConfig.START_DATE, DataConfig.END_DATE
    )
    
    # Train models
    training_results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    for name, model in trainer.models.items():
        evaluator.add_model(model, name)
    
    evaluation_results = evaluator.evaluate_all(X_test, y_test)
    
    # Print results
    comparison_df = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save models and results
    trainer.save_all_models()
    evaluator.save_results(Path(DataConfig.MODELS_DIR) / "evaluation_results.json")
    comparison_df.to_csv(Path(DataConfig.MODELS_DIR) / "model_comparison.csv", index=False)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
