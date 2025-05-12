import mlflow
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import Generator, Optional
from datetime import datetime
import logging
from scipy import sparse
import torch

logger = logging.getLogger(__name__)

class StreamingTrainer:
    def __init__(self, config: dict):
        """
        Initialise l'entraîneur avec une configuration complète.
        
        Args:
            config: Doit contenir:
                   - model: type et params
                   - training: batch_size
                   - mlflow: (optionnel)
        """
        self.config = config
        self._validate_config()
        self._init_model()
        self.n_samples_seen = 0
        logger.info(f"Trainer initialisé avec modèle {config['model']['type']}")

    def _validate_config(self):
        """Valide la configuration requise"""
        if 'model' not in self.config:
            raise ValueError("Section 'model' manquante dans la configuration")
        if 'type' not in self.config['model']:
            raise ValueError("Clé 'model.type' manquante")
        if 'params' not in self.config['model']:
            self.config['model']['params'] = {}
            logger.warning("Aucun paramètre spécifié pour le modèle")

    def _init_model(self):
        """Initialise le modèle selon la configuration"""
        model_type = self.config['model']['type']
        params = self.config['model']['params']
        
        if model_type == 'SGDClassifier':
            self.model = SGDClassifier(**params)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

    def partial_fit(self, X, y, classes: Optional[list] = None):
        """Entraînement incrémental"""
        try:
            if sparse.issparse(X):
                X = X.toarray()
            self.model.partial_fit(X, y, classes=classes)
            self.n_samples_seen += len(y)
            logger.debug(f"Entraîné sur {len(y)} nouveaux échantillons")
        except Exception as e:
            logger.error(f"Erreur d'entraînement: {str(e)}")
            raise

    def save(self, path: str = None):
        """Sauvegarde le modèle"""
        save_path = Path(path or self.config.get('paths', {}).get('model', 'artifacts/model.joblib'))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"Modèle sauvegardé à {save_path}")

    def get_model_version(self):
        """Version simplifiée du modèle"""
        return f"{self.config['model']['type']}-v1"