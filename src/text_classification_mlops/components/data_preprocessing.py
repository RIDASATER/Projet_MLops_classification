from typing import Iterator, List
import yaml
from text_processing import AdvancedTextCleaner

class BatchProcessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.cleaner = AdvancedTextCleaner()

    def process_batches(self, text_stream: Iterator[str]) -> Iterator[List[str]]:
        batch = []
        for text in text_stream:
            batch.append(text)
            if len(batch) >= self.config['processing']['batch_size']:
                yield self.cleaner.process_batch(batch)
                batch = []
        if batch:
            yield self.cleaner.process_batch(batch)