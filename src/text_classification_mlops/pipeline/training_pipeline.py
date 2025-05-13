import logging
import mlflow
import yaml
from pathlib import Path
import sys
import os
import time
import numpy as np
from contextlib import nullcontext
from typing import Dict, Any
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil

# Configuration for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

def load_config(path: str = None) -> Dict[str, Any]:
    """Load and merge all configurations with enhanced error handling"""
    try:
        if path is None:
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', 'configs', 'training_config.yaml')
        
        logger.info(f"Loading configuration from {path}")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set default SVM config if not present
        if 'svm' not in config.get('models', {}):
            config['models']['svm'] = {
                'kernel': 'linear',
                'C': 1.0,
                'max_iter': 500,
                'cache_size': 1000
            }
            
        return config
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise ConfigError(f"Configuration error: {str(e)}")

def log_system_resources():
    """Log system resources before training"""
    mem = psutil.virtual_memory()
    logger.info(f"Memory Available: {mem.available / (1024**3):.2f} GB")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")

def create_model(model_type: str, config: Dict[str, Any]):
    """Create model with SVM-specific optimizations"""
    model_params = config['models'].get(model_type, {})
    
    if model_type == 'sgd':
        return SGDClassifier(
            loss="log_loss",
            penalty=model_params.get('penalty', 'l2'),
            alpha=model_params.get('alpha', 0.0001),
            max_iter=model_params.get('max_iter', 1000),
            random_state=42
        )
    elif model_type == 'logistic_regression':
        return LogisticRegression(
            penalty=model_params.get('penalty', 'l2'),
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 1000),
            solver=model_params.get('solver', 'lbfgs'),
            random_state=42
        )
    elif model_type == 'svm':
        logger.info("Creating SVM model with optimized settings")
        return LinearSVC(  # Using LinearSVC instead of SVC for better performance
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 500),
            random_state=42,
            dual=False  # Better for n_samples > n_features
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive metrics"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

def train_model(model_type: str, pipeline, X_train, y_train):
    """Train model with progress logging"""
    logger.info(f"Training {model_type} model...")
    start_time = time.time()
    
    if model_type == 'svm':
        # For SVM, train on subset if data is large
        if len(X_train) > 10000:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, 
                train_size=10000, 
                random_state=42
            )
            logger.info("Using subset of 10,000 samples for SVM training")
    
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"{model_type} trained in {train_time:.2f} seconds")
    return pipeline, train_time

def log_to_mlflow(model_type: str, pipeline, metrics, train_time, X_test):
    """Log all results to MLflow"""
    with mlflow.start_run(run_name=f"{model_type}-run", nested=True):
        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "training_time": train_time
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        signature = infer_signature(X_test[:1], pipeline.predict(X_test[:1]))
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,
            registered_model_name=f"TextClassifier_{model_type}"
        )
        logger.info(f"Logged {model_type} model to MLflow")

def run_pipeline():
    """Main training pipeline with enhanced SVM support"""
    try:
        # Load config and check resources
        config = load_config()
        log_system_resources()
        
        # Load data
        df = pd.read_csv('data/raw/Dataset.csv')
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must contain 'review' and 'sentiment' columns")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['review'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment_name = "text-classification-comparison"
        mlflow.set_experiment(experiment_name)
        
        # Train all models
        for model_type in ['sgd', 'logistic_regression', 'svm']:
            try:
                logger.info(f"\n{'='*50}\nTraining {model_type}\n{'='*50}")
                
                # Create pipeline
                pipeline = make_pipeline(
                    TfidfVectorizer(),
                    create_model(model_type, config)
                )
                
                # Train and evaluate
                pipeline, train_time = train_model(model_type, pipeline, X_train, y_train)
                metrics = evaluate_model(pipeline, X_test, y_test)
                
                # Log results
                log_to_mlflow(model_type, pipeline, metrics, train_time, X_test)
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                mlflow.log_text(str(e), f"{model_type}_error.log")
                continue
                
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Training pipeline completed")

if __name__ == "__main__":
    run_pipeline()