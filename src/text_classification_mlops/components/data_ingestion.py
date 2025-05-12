# src/text_classification_mlops/components/data_ingestion.py
import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataStreamer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise avec un dictionnaire de configuration directement
        
        Args:
            config: Dictionnaire contenant les chemins nécessaires
                   Doit contenir 'raw_data', 'processed_data', etc.
        """
        self.config = config
        self.validate_config()
        
    def validate_config(self):
        """Vérifie que la configuration contient les clés nécessaires"""
        required_keys = ['raw_data', 'processed_data']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Configuration manquante: {key}")

    def stream_batches(self, batch_size: int = None):
        """
        Lit le fichier source et génère des batches de données
        
        Args:
            batch_size: Taille des lots (optionnel, peut être dans config)
        """
        batch_size = batch_size or self.config.get('batch_size', 512)
        
        try:
            # Lecture du fichier source
            for batch in pd.read_csv(
                self.config['raw_data'],
                chunksize=batch_size
            ):
                yield batch
                
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des données: {str(e)}")
            raise