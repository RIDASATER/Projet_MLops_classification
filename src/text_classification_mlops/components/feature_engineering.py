from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Generator, List
import yaml
import joblib
from pathlib import Path
from utils import load_config

class StreamingVectorizer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config('config')
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['features']['max_features'],
            tokenizer=lambda x: x.split() if isinstance(x, str) else x,
            preprocessor=lambda x: ' '.join(x) if isinstance(x, list) else x,
            ngram_range=tuple(self.config['features']['ngram_range'])
        )

    def partial_fit(self, texts: List[str]):
        """Adapté pour les tokens ou textes bruts"""
        self.vectorizer.fit(texts)

    def transform_stream(self, text_stream: Generator) -> Generator:
        for batch in text_stream:
            yield self.vectorizer.transform(batch)

    def save(self):
        path = Path(self.config['data']['vectorizer_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)