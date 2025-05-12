import re
import spacy
from typing import List, Generator, Optional, Dict, Any
from functools import lru_cache
from transformers import AutoTokenizer

class TextPreprocessor:
    """Classe de prétraitement de texte compatible avec les deux styles de configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le prétraitement avec configuration.
        
        Args:
            config: Doit contenir soit 'use_huggingface', soit 'framework'
        """
        self.config = self._normalize_config(config)
        self.nlp = self._init_spacy() if not self.config['use_huggingface'] else None
        self.tokenizer = self._init_tokenizer() if self.config['use_huggingface'] else None

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapte la configuration aux deux formats possibles"""
        normalized = {
            'use_huggingface': config.get('use_huggingface', False),
            'spacy': config.get('spacy', {}),
            'huggingface': config.get('huggingface', {}),
            'cleaning': config.get('cleaning', {}),
            'custom_stopwords': config.get('custom_stopwords', []),
            'performance': config.get('performance', {})
        }
        
        # Compatibilité avec l'ancien format 'framework'
        if 'framework' in config:
            normalized['use_huggingface'] = (config['framework']['active'] == 'huggingface')
            if 'spacy' in config['framework']:
                normalized['spacy'] = config['framework']['spacy']
            if 'huggingface' in config['framework']:
                normalized['huggingface'] = config['framework']['huggingface']
        
        return normalized

    @lru_cache(maxsize=1)
    def _init_spacy(self):
        """Initialise le modèle SpaCy avec gestion des erreurs."""
        try:
            nlp = spacy.load(
                self.config['spacy'].get('model_name', 'fr_core_news_md'),
                disable=self.config['spacy'].get('disable', [])
            )
            
            # Ajout des stopwords personnalisés
            for word in self.config.get('custom_stopwords', []):
                nlp.Defaults.stop_words.add(word)
                nlp.vocab[word].is_stop = True
                
            return nlp
        except OSError as e:
            raise ImportError(
                f"Modèle SpaCy {self.config['spacy'].get('model_name')} non trouvé. "
                f"Installez-le avec: python -m spacy download {self.config['spacy'].get('model_name')}"
            ) from e

    def _init_tokenizer(self) -> Optional[AutoTokenizer]:
        """Initialise le tokenizer HuggingFace."""
        return AutoTokenizer.from_pretrained(
            self.config['huggingface'].get('model_name', 'camembert-base'),
            use_fast=self.config['huggingface'].get('tokenizer_args', {}).get('use_fast', True)
        )

    def normalize_text(self, text: str) -> str:
        """Nettoyage de base avant traitement NLP."""
        text = text.strip()
        cleaning = self.config.get('cleaning', {})
        
        if cleaning.get('remove_urls', True):
            text = re.sub(r'http\S+|www\S+', '', text)
        if cleaning.get('remove_mentions', True):
            text = re.sub(r'@\w+', '', text)
        if cleaning.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        if cleaning.get('remove_numbers', True):
            text = re.sub(r'\d+', '', text)
        if cleaning.get('whitespace', True):
            text = re.sub(r'\s+', ' ', text)
        if cleaning.get('lowercase', True):
            text = text.lower()
            
        return text

    def process_single(self, text: str) -> str:
        """Traite un texte selon le framework configuré."""
        text = self.normalize_text(text)
        return self._tokenize_with_hf(text) if self.config['use_huggingface'] else self._process_with_spacy(text)

    def _process_with_spacy(self, text: str) -> str:
        """Pipeline complet avec SpaCy."""
        doc = self.nlp(text)
        min_len = self.config.get('performance', {}).get('min_token_length', 2)
        
        return " ".join(
            token.lemma_ 
            for token in doc
            if not token.is_stop 
            and not token.is_punct
            and len(token.lemma_) >= min_len
        )

    def _tokenize_with_hf(self, text: str) -> List[str]:
        """Tokenisation avec Hugging Face."""
        return self.tokenizer.tokenize(text)

    def process_batch(self, texts: List[str]) -> Generator[List[str], None, None]:
        """Traitement par lots optimisé."""
        batch_size = self.config.get('performance', {}).get('batch_size', 256)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.config['use_huggingface']:
                yield [self._tokenize_with_hf(text) for text in batch]
            else:
                yield [self._process_with_spacy(text) for text in batch]

    def __call__(self, text: str) -> str:
        """Interface fonctionnelle pour le prétraitement."""
        return self.process_single(text)