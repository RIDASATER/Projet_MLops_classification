import pandas as pd
from pathlib import Path

def load_data(config_path: str) -> pd.DataFrame:
    """Charge les données depuis le chemin spécifié dans config.yaml"""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_path = Path(config['data']['raw_path'])
    return pd.read_csv(data_path)