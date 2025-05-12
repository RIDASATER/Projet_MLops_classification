import logging
import mlflow
import yaml
from pathlib import Path
import sys
import os
import numpy as np
from contextlib import nullcontext
from typing import Dict, Any
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd  # Import pandas to load CSV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


# Configuration for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class TextClassificationWrapper(PythonModel):
    """MLflow wrapper for text classification model"""
    def __init__(self, model, vectorizer, processor):
        self.model = model
        self.vectorizer = vectorizer
        self.processor = processor

    def predict(self, context, model_input):
        processed_text = self.processor.process_batch(model_input.iloc[:, 0].tolist())
        features = self.vectorizer.transform(processed_text)
        return self.model.predict(features)

def load_config(path: str = None) -> Dict[str, Any]:
    """Load and merge all configurations with enhanced error handling"""
    try:
        if path is None:
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', 'configs', 'training_config.yaml')
        
        logger.info(f"Loading configuration from {path}")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {path}")
        raise ConfigError(f"Configuration not found: {path}")
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {str(e)}")
        raise ConfigError(f"YAML parsing error: {str(e)}")

def ensure_paths_exist(config: Dict[str, Any]) -> None:
    """Create necessary directories with improved error handling"""
    for key, path in config.get('paths', {}).items():
        directory = Path(path)
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created: {path}")
            except Exception as e:
                logger.error(f"Error creating directory {path}: {str(e)}")
                raise

def initialize_mlflow(config: Dict[str, Any]):
    """Initialize MLflow with error handling"""
    mlflow_config = config.get('mlflow', {})
    if not mlflow_config.get('tracking_uri'):
        logger.warning("MLflow disabled - no tracking will be recorded")
        return nullcontext()
    
    try:
        mlflow.set_tracking_uri("http://localhost:5000")  # Or any other local path
        experiment_name = mlflow_config.get('experiment_name', 'text-classification')
        mlflow.set_experiment(experiment_name)
        
        run_name = mlflow_config.get('run_name', f"{config['model']['type']}-streaming")
        run = mlflow.start_run(run_name=run_name)
        
        mlflow.set_tags({
            "project": "text-classification",
            "team": "rida",
            "framework": config['text_processing']['framework']['active'],
            "streaming": "true",
            "data_version": config.get('data_version', '1.0')
        })
        
        logger.info(f"MLflow enabled - Experiment: {experiment_name}, Run: {run_name}")
        
        # Log configurations
        for config_file in ['config.yaml', 'params.yaml', 'training_config.yaml']:
            config_path = Path(f"configs/{config_file}")
            if config_path.exists():
                mlflow.log_artifact(str(config_path), "configs")
        
        return run
    except Exception as e:
        logger.error(f"MLflow initialization failed: {str(e)}")
        return nullcontext()

def log_batch_metrics(trainer, X_batch, y_batch, batch_idx):
    """Log metrics for a training batch"""
    if not mlflow.active_run():
        return
        
    try:
        predictions = trainer.predict(X_batch)
        accuracy = np.mean(predictions == y_batch)
        
        mlflow.log_metrics({
            'batch/accuracy': accuracy,
            'batch/loss': trainer.current_loss,
            'batch/size': len(y_batch)
        }, step=trainer.n_samples_seen)
        
        if batch_idx % 10 == 0:  # Every 10 batches
            mlflow.log_metric('cumulative_samples', trainer.n_samples_seen)
    except Exception as e:
        logger.warning(f"Metrics logging failed: {str(e)}")

def run_pipeline():
    """Execute the complete streaming training pipeline."""
    try:
        # Load and validate configuration
        config = load_config()
        ensure_paths_exist(config)
        
        # Initialize MLflow
        with initialize_mlflow(config) as mlflow_context:
            # Load your dataset
            df = pd.read_csv('data/raw/Dataset.csv')  # Update with your local path
            # Ensure the dataset has the necessary columns
            if 'review' not in df.columns or 'sentiment' not in df.columns:
                raise ValueError(f"Dataset must contain 'review' and 'sentiment' columns")

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
            
            # Create a pipeline with TfidfVectorizer and SGDClassifier
            model_pipeline = make_pipeline(
                TfidfVectorizer(),  # Convert text data to numeric using TF-IDF
                SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001)
            )
            
            # Train the model
            model_pipeline.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_pred = model_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            if mlflow.active_run():
                mlflow.log_metric("test_accuracy", accuracy)
                
                # Log the model
                signature = infer_signature(X_test[:1], model_pipeline.predict(X_test[:1]))
                mlflow.sklearn.log_model(
                    model_pipeline,
                    "model",
                    signature=signature,
                    registered_model_name="TextClassifier"
                )
                
                logger.info(f"Model registered in MLflow with run ID: {mlflow.active_run().info.run_id}")
                
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("Pipeline execution completed")
        
if __name__ == "__main__":
    run_pipeline()
