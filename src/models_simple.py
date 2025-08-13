"""
Simplified model training that works with the simple feature engineering
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from config import ModelConfig
from src.feature_engineering_simple import SimpleFeatureEngineer

logger = logging.getLogger(__name__)

class SimpleModelTrainer:
    """Simple model trainer for wildfire prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_engineer = SimpleFeatureEngineer()
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Load and prepare data for training"""
        logger.info("Preparing data for training...")
        
        X, y, feature_columns, df = self.feature_engineer.prepare_features()
        
        if X is None:
            raise ValueError("No data available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Fire occurrence in training: {y_train.mean():.4f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"Random Forest CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return rf_model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression classifier"""
        logger.info("Training Logistic Regression model...")
        
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        lr_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"Logistic Regression CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return lr_model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train simple neural network if TensorFlow is available"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network training")
            return None
        
        logger.info("Training Neural Network model...")
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Train with early stopping
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Get final validation metrics
        val_loss, val_accuracy, val_auc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Neural Network - Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model"""
        logger.info(f"Evaluating {model_name}...")
        
        if TENSORFLOW_AVAILABLE and hasattr(model, 'predict_proba') == False and hasattr(model, 'predict'):
            # TensorFlow model
            y_pred_proba = model.predict(X_test).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Scikit-learn model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  ROC-AUC Score: {auc_score:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"  Precision: {report['1']['precision']:.4f}")
        logger.info(f"  Recall: {report['1']['recall']:.4f}")
        logger.info(f"  F1-Score: {report['1']['f1-score']:.4f}")
        
        return {
            'auc': auc_score,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score']
        }
    
    def save_models(self, feature_columns):
        """Save trained models and metadata"""
        logger.info("Saving models...")
        
        # Save scikit-learn models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':
                model_path = self.models_dir / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_name} to {model_path}")
        
        # Save TensorFlow model separately
        if 'neural_network' in self.models and self.models['neural_network'] is not None:
            nn_path = self.models_dir / "neural_network.h5"
            self.models['neural_network'].save(nn_path)
            logger.info(f"Saved neural_network to {nn_path}")
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = self.models_dir / "feature_columns.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        logger.info(f"Saved feature columns to {features_path}")
        
        # Save metadata
        metadata = {
            'model_types': list(self.models.keys()),
            'n_features': len(feature_columns),
            'feature_names': feature_columns,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.models_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def train_all_models(self):
        """Train all models and evaluate performance"""
        logger.info("Starting model training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data()
        
        # Train models
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['neural_network'] = self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        results = {}
        for model_name, model in self.models.items():
            if model is not None:
                results[model_name] = self.evaluate_model(model, X_test, y_test, model_name)
        
        # Save models
        self.save_models(feature_columns)
        
        # Print summary
        logger.info("="*60)
        logger.info("MODEL TRAINING SUMMARY")
        logger.info("="*60)
        
        for model_name, metrics in results.items():
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  ROC-AUC: {metrics['auc']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            logger.info("-" * 40)
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        logger.info(f"Best performing model: {best_model}")
        logger.info(f"Best ROC-AUC: {results[best_model]['auc']:.4f}")
        
        logger.info("="*60)
        logger.info("Model training completed successfully!")
        
        return results

def main():
    """Main training pipeline"""
    logger.info("Starting simple model training pipeline")
    
    trainer = SimpleModelTrainer()
    results = trainer.train_all_models()
    
    return results

if __name__ == "__main__":
    main()
