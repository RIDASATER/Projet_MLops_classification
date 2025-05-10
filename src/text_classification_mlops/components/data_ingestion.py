import pandas as pd
from typing import Generator, Dict
from pathlib import Path
import yaml
from utils import validate_text_columns

class DataStreamer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.raw_path = Path(self.config['data']['raw_path'])
        
    def stream_records(self) -> Generator[Dict, None, None]:
        """Générateur qui produit des dictionnaires avec validation"""
        for chunk in pd.read_csv(self.raw_path, chunksize=self.config['processing']['batch_size']):
            validate_text_columns(chunk)  # Validation ajoutée
            for _, row in chunk.iterrows():
                yield {
                    'text': row['text'],
                    'sentiment': row['sentiment'],
                    'metadata': {k: v for k, v in row.items() if k not in ['text', 'sentiment']}
                }

    def get_batches(self) -> Generator[pd.DataFrame, None, None]:
        """Version avec validation des colonnes"""
        for chunk in pd.read_csv(self.raw_path, chunksize=self.config['processing']['batch_size']):
            validate_text_columns(chunk)
            yield chunk