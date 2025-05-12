from typing import Iterator, List
from pathlib import Path
import yaml
from text_classification_mlops.components.text_processing import TextPreprocessor
class BatchProcessor:
    def __init__(self, config: dict):
        """
        Initialise avec la configuration de traitement.
        
        Args:
            config: Doit contenir la configuration text_processing et performance
        """
        self.preprocessor = TextPreprocessor(config)
        self.batch_size = config.get('performance', {}).get('batch_size', 256)

    def process_stream(self, text_stream: Iterator[str]) -> Iterator[List[str]]:
        """Traite un flux de textes en lots."""
        batch = []
        for text in text_stream:
            batch.append(text)
            if len(batch) >= self.batch_size:
                yield self.preprocessor.process_batch(batch)
                batch = []
        if batch:
            yield self.preprocessor.process_batch(batch)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Traite une liste de textes directement."""
        return self.preprocessor.process_batch(texts)