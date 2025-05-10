# text_processing.py
import spacy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Generator
import re
from functools import lru_cache
import logging
from pathlib import Path
import yaml

class AdvancedTextCleaner:
    """Pipeline de prétraitement de texte optimisé pour le traitement par lots"""
    
    def __init__(self, config_path: str = "configs/text_processing_config.yaml"):
        """
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self._init_logging()
        self.nlp = self._load_spacy_model()
        
    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration depuis un fichier YAML"""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Erreur de chargement de la config: {e}")
            raise

    def _init_logging(self):
        """Configure le système de logging"""
        logging.basicConfig(
            level=self.config.get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        """Charge le modèle SpaCy avec cache"""
        try:
            return spacy.load(
                self.config.get('spacy_model', 'fr_core_news_sm'),
                disable=["parser", "ner", "lemmatizer"]
            )
        except OSError:
            raise ImportError("Modèle SpaCy non trouvé. Installez-le avec: "
                            "python -m spacy download fr_core_news_sm")

    def _preprocess_text(self, text: str) -> str:
        """Nettoyage de base avant le traitement NLP"""
        # 1. Suppression des URLs et mentions
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        # 2. Normalisation des espaces
        return ' '.join(text.strip().lower().split())

    def process(self, text: str) -> str:
        """
        Traitement complet d'un texte unique
        Args:
            text: Texte brut en entrée
        Returns:
            Texte nettoyé et lemmatisé
        """
        try:
            cleaned = self._preprocess_text(text)
            doc = self.nlp(cleaned)
            return " ".join(
                token.lemma_ 
                for token in doc 
                if not token.is_stop and token.is_alpha
            )
        except Exception as e:
            logging.error(f"Erreur de traitement du texte: {e}")
            return ""

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Traitement parallélisé d'une liste de textes
        Args:
            texts: Liste de textes bruts
        Returns:
            Liste de textes traités
        """
        with ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        ) as executor:
            return list(executor.map(self.process, texts))

    def stream_processing(self, text_stream: Generator[str, None, None]) -> Generator[str, None, None]:
        """
        Traitement en flux continu pour les gros volumes
        Args:
            text_stream: Générateur de textes bruts
        Yields:
            Textes traités un par un
        """
        for text in text_stream:
            yield self.process(text)

    def save_processing_stats(self, output_path: str):
        """
        Sauvegarde les statistiques de traitement
        Args:
            output_path: Chemin de sauvegarde
        """
        stats = {
            'spacy_model': self.config.get('spacy_model'),
            'processing_date': datetime.now().isoformat()
        }
        with open(output_path, 'w') as f:
            yaml.dump(stats, f)