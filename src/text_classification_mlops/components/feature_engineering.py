from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Generator, List, Union
import logging
import joblib
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class StreamingVectorizer:
    """Vectoriseur de texte avec support pour le streaming et gestion de configuration robuste"""
    
    DEFAULT_CONFIG = {
        'features': {
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2),
            'tokenizer': None,
            'preprocessor': None
        },
        'paths': {
            'vectorizer': 'artifacts/vectorizer.joblib'
        }
    }

    def __init__(self, config: Union[str, dict] = None):
        """
        Initialise le vectoriseur.
        
        Args:
            config: Chemin vers un fichier YAML ou dictionnaire de configuration.
                   Si None, utilise les valeurs par défaut.
        """
        self.config = self._load_config(config)
        self._init_vectorizer()
        logger.info("Vectoriseur initialisé avec la configuration: %s", self.config['features'])

    def _load_config(self, config) -> dict:
        """Charge et fusionne la configuration"""
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Erreur chargement config, utilisation valeurs par défaut: %s", str(e))
                loaded_config = {}
        elif isinstance(config, dict):
            loaded_config = config
        else:
            loaded_config = {}

        # Fusion profonde des configurations
        merged_config = {
            'features': {**self.DEFAULT_CONFIG['features'], **loaded_config.get('features', {})},
            'paths': {**self.DEFAULT_CONFIG['paths'], **loaded_config.get('paths', {})}
        }
        return merged_config

    def _init_vectorizer(self):
        """Initialise le vectoriseur TF-IDF avec la configuration"""
        features_cfg = self.config['features']
        
        self.vectorizer = TfidfVectorizer(
            max_features=features_cfg['max_features'],
            min_df=features_cfg['min_df'],
            max_df=features_cfg['max_df'],
            ngram_range=tuple(features_cfg['ngram_range']),
            tokenizer=features_cfg['tokenizer'] or (lambda x: x.split() if isinstance(x, str) else x),
            preprocessor=features_cfg['preprocessor'] or (lambda x: ' '.join(x) if isinstance(x, list) else x)
        )

    def partial_fit(self, texts: List[str]):
        """Adapte le vectoriseur à un batch de textes"""
        try:
            self.vectorizer.fit(texts)
            logger.debug("Vectoriseur mis à jour avec %d échantillons", len(texts))
        except Exception as e:
            logger.error("Erreur pendant partial_fit: %s", str(e))
            raise

    def transform(self, texts: List[str]):
        """Transforme une liste de textes en matrice de features"""
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            logger.error("Erreur pendant transform: %s", str(e))
            raise

    def transform_stream(self, text_stream: Generator[List[str], None, None]) -> Generator:
        """Transforme un flux de batches de textes"""
        for batch in text_stream:
            try:
                yield self.transform(batch)
            except Exception as e:
                logger.error("Erreur dans transform_stream: %s", str(e))
                raise

    def save(self, path: str = None):
        """Sauvegarde le vectoriseur sur disque"""
        save_path = Path(path or self.config['paths']['vectorizer'])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self.vectorizer, save_path)
            logger.info("Vectoriseur sauvegardé à %s", save_path)
        except Exception as e:
            logger.error("Erreur sauvegarde vectoriseur: %s", str(e))
            raise

    @classmethod
    def load(cls, path: str, config: Union[str, dict] = None):
        """Charge un vectoriseur depuis disque"""
        try:
            vectorizer = joblib.load(path)
            instance = cls(config)
            instance.vectorizer = vectorizer
            logger.info("Vectoriseur chargé depuis %s", path)
            return instance
        except Exception as e:
            logger.error("Erreur chargement vectoriseur: %s", str(e))
            raise