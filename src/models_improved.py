"""
Improved model training with better neural network architecture and hyperparameters
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from config import ModelConfig
from src.feature_engineering_simple import SimpleFeatureEngineer

logger = logging.getLogger(__name__)

class ImprovedModelTrainer:
    """Improved model trainer with better neural network and hyperparameters"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_engineer = SimpleFeatureEngineer()
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Load and prepare data for training with better preprocessing"""
        logger.info("Preparing data for training...")
        
        X, y, feature_columns, df = self.feature_engineer.prepare_features()
        
        if X is None:
            raise ValueError("No data available for training")
        
        # Handle class imbalance - oversample minority class slightly
        from sklearn.utils import resample
        
        # Combine features and target
        data_combined = pd.concat([X, y], axis=1)
        
        # Separate majority and minority classes
        majority = data_combined[data_combined.iloc[:, -1] == 0]
        minority = data_combined[data_combined.iloc[:, -1] == 1]
        
        logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
        
        # Oversample minority class to reduce imbalance (but not fully balance)
        minority_upsampled = resample(minority,
                                    replace=True,
                                    n_samples=min(len(majority) // 3, len(minority) * 3),
                                    random_state=42)
        
        # Combine majority and upsampled minority
        balanced_data = pd.concat([majority, minority_upsampled])
        
        # Separate features and target
        X_balanced = balanced_data.iloc[:, :-1]
        y_balanced = balanced_data.iloc[:, -1]
        
        logger.info(f"Balanced class distribution: {y_balanced.value_counts().to_dict()}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
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
        """Train improved Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        # Cross-validation with stratified folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"Random Forest CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return rf_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting model...")
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(gb_model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"Gradient Boosting CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return gb_model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train improved Logistic Regression classifier"""
        logger.info("Training Logistic Regression model...")
        
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            C=0.1,  # L2 regularization
            solver='lbfgs'
        )
        
        lr_model.fit(X_train, y_train)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"Logistic Regression CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return lr_model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train improved neural network if TensorFlow is available"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network training")
            return None
        
        logger.info("Training Improved Neural Network model...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(y_train), 
                                           y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Improved architecture
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), 
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Improved optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'precision', 'recall']
        )
        
        # Improved callbacks
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train with better parameters
        logger.info("Training neural network with improved architecture...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Get final validation metrics
        val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Neural Network - Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"Neural Network - Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Test prediction to ensure it's working
        test_pred = model.predict(X_test[:5], verbose=0)
        logger.info(f"Sample predictions: {test_pred.flatten()}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model with detailed metrics"""
        logger.info(f"Evaluating {model_name}...")
        
        if TENSORFLOW_AVAILABLE and hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
            # TensorFlow model
            y_pred_proba = model.predict(X_test, verbose=0).ravel()
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
        
        # Prediction distribution
        logger.info(f"  Prediction range: {y_pred_proba.min():.6f} to {y_pred_proba.max():.6f}")
        logger.info(f"  Mean prediction: {y_pred_proba.mean():.6f}")
        
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
            'training_date': pd.Timestamp.now().isoformat(),
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
        
        metadata_path = self.models_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def train_all_models(self):
        """Train all models and evaluate performance"""
        logger.info("Starting improved model training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data()
        
        # Train models
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['gradient_boosting'] = self.train_gradient_boosting(X_train, y_train)
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
        logger.info("="*70)
        logger.info("IMPROVED MODEL TRAINING SUMMARY")
        logger.info("="*70)
        
        for model_name, metrics in results.items():
            logger.info(f"{model_name.upper().replace('_', ' ')}:")
            logger.info(f"  ROC-AUC: {metrics['auc']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            logger.info("-" * 50)
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        logger.info(f"🏆 Best performing model: {best_model}")
        logger.info(f"🏆 Best ROC-AUC: {results[best_model]['auc']:.4f}")
        
        logger.info("="*70)
        logger.info("🎉 Improved model training completed successfully!")
        
        return results

def main():
    """Main training pipeline"""
    logger.info("Starting improved model training pipeline")
    
    trainer = ImprovedModelTrainer()
    results = trainer.train_all_models()
    
    return results

if __name__ == "__main__":
    main()
