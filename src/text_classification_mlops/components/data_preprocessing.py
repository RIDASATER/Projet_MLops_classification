from typing import Iterator
import yaml
from text_processing import TextPreprocessor

class BatchProcessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.preprocessor = TextPreprocessor()
        
    def process_stream(self, text_stream: Iterator[str]) -> Iterator[List[str]]:
        """Adapté pour le traitement en flux"""
        batch = []
        for text in text_stream:
            batch.append(text)
            if len(batch) >= self.config['processing']['batch_size']:
                yield from self.preprocessor.process_batch(batch)
                batch = []
        if batch:
            yield from self.preprocessor.process_batch(batch)