from data_ingestion import DataStreamer
from data_preprocessing import BatchProcessor
from feature_engineering import StreamingVectorizer
from model_trainer import StreamingTrainer
import yaml
import mlflow
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    try:
        # Chargement des configurations
        main_config = yaml.safe_load(open("configs/config.yaml"))
        mlflow.set_tracking_uri(main_config['mlflow']['tracking_uri'])
        mlflow.set_experiment(main_config['mlflow']['experiment_name'])

        # Initialisation des composants
        logger.info("Initialisation des composants...")
        streamer = DataStreamer()
        processor = BatchProcessor()
        vectorizer = StreamingVectorizer()
        trainer = StreamingTrainer()

        # Préparation des données
        logger.info("Préparation des données...")
        sample_batch = next(streamer.get_batches())
        vectorizer.partial_fit(sample_batch['text'].tolist())

        text_stream = (text for batch in streamer.get_batches() for text in batch['text'])
        clean_batches = processor.process_batches(text_stream)
        X_stream = vectorizer.transform_stream(clean_batches)
        y_stream = (batch['sentiment'].values for batch in streamer.get_batches())

        # Entraînement
        logger.info("Début de l'entraînement...")
        trainer.online_fit(X_stream, y_stream)
        vectorizer.save()
        logger.info("Pipeline exécuté avec succès!")

    except Exception as e:
        logger.error(f"Erreur dans le pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()