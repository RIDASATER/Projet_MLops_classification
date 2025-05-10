import spacy
import re
from typing import List, Generator
from functools import lru_cache
from transformers import AutoTokenizer
import yaml

class TextPreprocessor:
    def __init__(self, config_path: str = "configs/text_processing_config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.nlp = self._load_spacy_model()
        self.tokenizer = self._load_hf_tokenizer() if self.config.get('use_huggingface') else None

    @lru_cache(maxsize=None)
    def _load_spacy_model(self):
        nlp = spacy.load(
            self.config['spacy']['model_name'],
            disable=self.config['spacy']['disable']
        )
        # Ajout de règles personnalisées
        if self.config['custom_stopwords']:
            for word in self.config['custom_stopwords']:
                nlp.Defaults.stop_words.add(word)
        return nlp

    def _load_hf_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.config['huggingface']['model_name']
        )

    def _clean_text(self, text: str) -> str:
        """Nettoyage de base avant traitement NLP"""
        # Suppression URLs/mentions
        if self.config['cleaning']['remove_urls']:
            text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        # Normalisation
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower() if self.config['cleaning']['lowercase'] else text

    def spacy_process(self, text: str) -> str:
        """Pipeline complet avec SpaCy"""
        text = self._clean_text(text)
        doc = self.nlp(text)
        return " ".join(
            token.lemma_ 
            for token in doc
            if not token.is_stop 
            and not token.is_punct
            and len(token.lemma_) > 1
        )

    def hf_tokenize(self, text: str) -> List[str]:
        """Tokenisation avec Hugging Face"""
        text = self._clean_text(text)
        return self.tokenizer.tokenize(text)

    def process_batch(self, texts: List[str]) -> Generator[List[str], None, None]:
        """Traitement par lots optimisé"""
        batch_size = self.config['performance']['batch_size']
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.tokenizer:
                yield [self.hf_tokenize(text) for text in batch]
            else:
                yield [self.spacy_process(text) for text in batch]