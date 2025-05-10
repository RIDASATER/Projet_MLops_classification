from data_ingestion import DataStreamer
from data_preprocessing import BatchProcessor
from feature_engineering import StreamingVectorizer
from model_trainer import StreamingTrainer
from utils import load_config
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    try:
        # Chargement config
        config = load_config('config')
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
        # Initialisation
        streamer = DataStreamer()
        processor = BatchProcessor()
        vectorizer = StreamingVectorizer()
        trainer = StreamingTrainer()

        # 1. Entraînement du vectoriseur
        sample_batch = next(streamer.get_batches())
        vectorizer.partial_fit(sample_batch['text'].tolist())

        # 2. Flux de traitement
        text_stream = (text for batch in streamer.get_batches() for text in batch['text'])
        processed_stream = processor.process_stream(text_stream)
        
        # 3. Vectorisation et entraînement
        X_stream = vectorizer.transform_stream(processed_stream)
        y_stream = (batch['sentiment'].values for batch in streamer.get_batches())
        
        trainer.online_fit(X_stream, y_stream)
        vectorizer.save()

    except Exception as e:
        logger.error(f"Erreur: {str(e)}", exc_info=True)
        raise