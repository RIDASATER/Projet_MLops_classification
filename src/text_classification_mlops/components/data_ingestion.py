# data_ingestion.py
import pandas as pd
from pathlib import Path
from typing import Generator, Union
import yaml

class DataStreamer:
    """Charge les données en mode paresseux avec gestion des chemins config.yaml"""
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Charge la configuration depuis le fichier YAML"""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def stream_records(self, chunk_size: int = 512) -> Generator[dict, None, None]:
        """
        Générateur qui produit les enregistrements un par un
        Args:
            chunk_size: Nombre de lignes à lire à la fois (optimisation mémoire)
        Yields:
            Dictionnaires représentant chaque ligne du CSV
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.data_path}")

        for chunk in pd.read_csv(
            self.data_path,
            chunksize=chunk_size,
            dtype={'text': str, 'sentiment': 'category'}  # Optimisation mémoire
        ):
            for _, record in chunk.iterrows():
                yield record.to_dict()

    def get_batches(self, batch_size: int = 256) -> Generator[pd.DataFrame, None, None]:
        """
        Version améliorée avec vérification du fichier et typage
        Args:
            batch_size: Taille des lots à produire
        Yields:
            DataFrames contenant les lots de données
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.data_path}")

        return pd.read_csv(
            self.data_path,
            chunksize=batch_size,
            dtype={'text': str, 'sentiment': 'category'}
        )

# Fonction existante adaptée pour la rétro-compatibilité
def load_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Fonction legacy pour chargement complet (à utiliser pour les petits datasets)"""
    streamer = DataStreamer(config_path)
    return pd.concat(streamer.get_batches(batch_size=10_000))  # Gros chunks pour optimisation