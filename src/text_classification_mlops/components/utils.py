import yaml
from pathlib import Path
import pandas as pd

def load_config(config_name: str) -> dict:
    """Charge un fichier YAML avec validation"""
    path = Path("configs") / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config {config_name} introuvable")
    return yaml.safe_load(open(path))

def validate_text_columns(df: pd.DataFrame):
    """Valide la structure des données texte"""
    required = ['text', 'sentiment']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes requises manquantes: {missing}")