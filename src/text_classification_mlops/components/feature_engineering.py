from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Generator, List
import yaml
import joblib
from pathlib import Path

class StreamingVectorizer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )

    def partial_fit(self, texts: List[str]):
        self.vectorizer.fit(texts)

    def transform_stream(self, text_stream: Generator[List[str], None, None]) -> Generator:
        for text_batch in text_stream:
            yield self.vectorizer.transform(text_batch)

    def save(self):
        path = Path(self.config['data']['vectorizer_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)