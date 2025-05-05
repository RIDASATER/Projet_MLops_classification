import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées avec succès depuis {file_path}")
        return df
    except Exception as e:
        raise Exception(f"Erreur lors du chargement des données: {str(e)}")

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Sauvegarde les données dans un fichier CSV"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Données sauvegardées avec succès dans {file_path}")
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde des données: {str(e)}")

def load_config(config_path: str) -> dict:
    """Charge la configuration YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)