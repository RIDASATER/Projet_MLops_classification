import mlflow
import numpy as np
from sklearn.linear_model import SGDClassifier
from typing import Iterator, Dict, Any
from scipy import sparse
import time
from pathlib import Path
import yaml

class StreamingTrainer:
    """Classe optimisée pour l'entraînement en flux continu avec suivi avancé"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Args:
            config_path: Chemin vers la configuration d'entraînement
        """
        self.config = self._load_config(config_path)
        self.model = self._init_model()
        self.batch_count = 0
        self.metrics_history = {
            'accuracy': [],
            'processing_time': []
        }
        
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML"""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _init_model(self) -> SGDClassifier:
        """Initialise le modèle avec les paramètres de configuration"""
        return SGDClassifier(
            loss=self.config['model']['loss'],
            penalty=self.config['model']['penalty'],
            alpha=self.config['model']['alpha'],
            max_iter=self.config['model']['max_iter'],
            tol=self.config['model']['tol'],
            early_stopping=self.config['model']['early_stopping'],
            n_iter_no_change=self.config['model']['n_iter_no_change'],
            class_weight=self.config['model']['class_weight']
        )
    
    def online_fit(self,
                  X_stream: Iterator[sparse.csr_matrix],
                  y_stream: Iterator[np.ndarray]) -> None:
        """
        Entraînement incrémental avec suivi complet
        
        Args:
            X_stream: Flux de matrices creuses de features
            y_stream: Flux de tableaux numpy de labels
        """
        with mlflow.start_run(run_name="streaming_training"):
            self._log_config()
            start_time = time.time()
            
            for X_batch, y_batch in zip(X_stream, y_stream):
                batch_start = time.time()
                
                # Entraînement
                self.model.partial_fit(
                    X_batch,
                    y_batch,
                    classes=np.array(["positive", "neutral", "negative"])
                )
                
                # Calcul des métriques
                processing_time = time.time() - batch_start
                acc = self.model.score(X_batch, y_batch)
                
                # Mise à jour des historiques
                self.batch_count += 1
                self.metrics_history['accuracy'].append(acc)
                self.metrics_history['processing_time'].append(processing_time)
                
                # Logging périodique
                if self.batch_count % self.config['logging']['batch_interval'] == 0:
                    self._log_metrics()
            
            # Finalisation
            self._finalize_training(start_time)
    
    def _log_config(self) -> None:
        """Enregistre la configuration dans MLflow"""
        mlflow.log_params({
            'model_type': 'SGDClassifier',
            **self.config['model'],
            **self.config['logging']
        })
        mlflow.set_tag("training_mode", "streaming")
    
    def _log_metrics(self) -> None:
        """Log les métriques dans MLflow"""
        mlflow.log_metrics({
            'accuracy': np.mean(self.metrics_history['accuracy'][-10:]),
            'batch_processing_time': np.mean(self.metrics_history['processing_time'][-10:]),
            'cumulative_batches': self.batch_count
        }, step=self.batch_count)
    
    def _finalize_training(self, start_time: float) -> None:
        """Opérations finales d'entraînement"""
        # Sauvegarde du modèle
        mlflow.sklearn.log_model(
            self.model,
            "model",
            registered_model_name=self.config['model']['registered_name'],
            metadata={
                'training_duration': time.time() - start_time,
                'total_batches': self.batch_count,
                'avg_accuracy': np.mean(self.metrics_history['accuracy']),
                'avg_processing_time': np.mean(self.metrics_history['processing_time'])
            }
        )
        
        # Sauvegarde des artefacts supplémentaires
        self._save_artifacts()
    
    def _save_artifacts(self) -> None:
        """Sauvegarde des artefacts complémentaires"""
        # Exemple : sauvegarde des poids de features importants
        if hasattr(self.model, 'coef_'):
            feature_importance = np.argsort(np.abs(self.model.coef_))[::-1]
            np.save("artifacts/feature_importance.npy", feature_importance)
            mlflow.log_artifact("artifacts/feature_importance.npy")
