from .data_access.local_storage import load_data, save_data
from .logger import logging
import os

def load_dataset(config_path: str) -> pd.DataFrame:
    """Charge le dataset brut selon la configuration"""
    try:
        config = load_config(config_path)
        raw_data_path = config['data']['raw_data_path']
        
        logging.info(f"Chargement des données depuis {raw_data_path}")
        df = load_data(raw_data_path)
        
        logging.info(f"Données chargées avec succès. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Erreur dans data_ingestion: {str(e)}")
        raise e