import mlflow
import numpy as np
from sklearn.linear_model import SGDClassifier
from utils import load_config
from typing import Generator
from scipy import sparse
from transformers import AutoModel
import torch

class StreamingTrainer:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config = load_config('training')
        self.model = SGDClassifier(**self.config['model']['params'])
        if self.config['model'].get('use_embeddings'):
            self.embedding_model = AutoModel.from_pretrained(
                self.config['huggingface']['model_name']
            )

    def online_fit(self, X_stream: Generator, y_stream: Generator):
        with mlflow.start_run():
            mlflow.log_params(self.config['model']['params'])
            
            for X_batch, y_batch in zip(X_stream, y_stream):
                if hasattr(self, 'embedding_model'):
                    X_batch = self._generate_embeddings(X_batch)
                self.model.partial_fit(X_batch, y_batch, classes=["positive", "neutral", "negative"])
                # ... (log metrics)

    def _generate_embeddings(self, texts):
        inputs = self.embedding_model.tokenizer(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()