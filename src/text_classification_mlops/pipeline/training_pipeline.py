import logging
import mlflow
import yaml
from pathlib import Path
import sys
import os
from contextlib import nullcontext
from typing import Dict, Any

# Configuration des imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from text_classification_mlops.components.data_ingestion import DataStreamer
from text_classification_mlops.components.data_preprocessing import BatchProcessor
from text_classification_mlops.components.feature_engineering import StreamingVectorizer
from text_classification_mlops.components.model_trainer import StreamingTrainer
from text_classification_mlops.components.text_processing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Exception personnalisée pour les erreurs de configuration"""
    pass

def load_config(path: str = "configs/training_config.yaml"):
    ...
    """Charge et fusionne toutes les configurations avec gestion des erreurs améliorée"""
    config_files = {
        'global': 'configs/config.yaml',
        'model': 'configs/params.yaml',
        'training': 'configs/training_config.yaml',
        'text_processing': 'configs/text_processing_config.yaml',
        'schema': 'configs/schema.yaml'
    }
    config = {}
    for name, file_path in config_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config[name] = yaml.safe_load(f) or {}
                logger.debug(f"Configuration chargée depuis {file_path}")
        except FileNotFoundError:
            logger.warning(f"Fichier de configuration {file_path} non trouvé")
            config[name] = {}
        except yaml.YAMLError as e:
            logger.error(f"Erreur YAML dans {file_path}: {str(e)}")
            raise ConfigError(f"Erreur dans le fichier de configuration {file_path}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement de {file_path}: {str(e)}")
            raise ConfigError(f"Impossible de charger {file_path}")

    # Configuration par défaut critique
    defaults = {
        'paths': {
            'artifacts_dir': 'artifacts/',
            'vectorizer': 'artifacts/vectorizer.pkl',
            'model': 'artifacts/model.joblib'
        },
        'training': {
            'log_interval': 1000,
            'batch_size': 512
        },
        'text_processing': {
            'framework': {
                'active': 'spacy'
            }
        }
    }
    # Fusion structurée avec validation
    required_sections = {
        'schema': ['text_column', 'target_column'],
        'model': ['type', 'params']
    }
    
    merged_config = {
        'paths': {**defaults['paths'], **config['global'].get('paths', {})},
        'mlflow': config['global'].get('mlflow', {}),
        'model': config['model'],
        'training': {**defaults['training'], **config['training']},
        'text_processing': {**defaults['text_processing'], **config['text_processing']},
        'schema': config['schema']
    }

    # Validation des sections requises
    for section, required_keys in required_sections.items():
        if not merged_config.get(section):
            raise ConfigError(f"Section de configuration manquante: {section}")
        for key in required_keys:
            if key not in merged_config[section]:
                raise ConfigError(f"Clé de configuration manquante: {section}.{key}")

    return merged_config

def ensure_paths_exist(config: Dict[str, Any]) -> None:
    """Crée les dossiers nécessaires avec gestion d'erreurs améliorée"""
    required_paths = {
        'raw_data': config['paths']['raw_data'],
        'processed_data': config['paths']['processed_data'],
        'artifacts': config['paths']['artifacts_dir'],
        'vectorizer': Path(config['paths']['vectorizer']).parent
    }
    
    for name, path in required_paths.items():
        try:
            path_obj = Path(path)
            if name in ['raw_data', 'processed_data']:
                if not path_obj.exists():
                    raise FileNotFoundError(f"Fichier {name} introuvable: {path}")
            else:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Dossier {name} vérifié/créé: {path}")
        except Exception as e:
            logger.error(f"Erreur avec le chemin {name}: {str(e)}")
            raise

def initialize_mlflow(config: Dict[str, Any]):
    """Initialise MLflow avec gestion des erreurs"""
    mlflow_config = config.get('mlflow', {})
    if not mlflow_config.get('tracking_uri'):
        logger.warning("MLflow désactivé - aucun suivi ne sera enregistré")
        return nullcontext()
    
    try:
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'text-classification'))
        run = mlflow.start_run()
        logger.info("MLflow activé - suivi des expériences")
        
        # Log des configurations
        for config_file in ['config.yaml', 'params.yaml', 'training_config.yaml']:
            config_path = Path(f"configs/{config_file}")
            if config_path.exists():
                mlflow.log_artifact(str(config_path))
        
        return run
    except Exception as e:
        logger.error(f"Échec de l'initialisation MLflow: {str(e)}")
        return nullcontext()

def run_pipeline():
    """Exécute le pipeline complet de formation en streaming."""
    try:
        # Chargement et validation de la configuration
        config = load_config()
        ensure_paths_exist(config)
        
        # Initialisation MLflow
        with initialize_mlflow(config) as mlflow_context:
            # Initialisation des composants
            logger.info("Initialisation des composants du pipeline")
            components = {
                'streamer': DataStreamer(config['paths']),
                'processor': BatchProcessor(config['text_processing']),
                'vectorizer': StreamingVectorizer(config),
                'trainer': StreamingTrainer(config['model'])
            }
            
            # Entraînement initial
            logger.info("Phase d'initialisation du vectoriseur")
            first_batch = next(components['streamer'].stream_batches())
            texts = components['processor'].process_batch(
                first_batch[config['schema']['text_column']].tolist()
            )
            components['vectorizer'].partial_fit(texts)
            
            # Pipeline de streaming principal
            logger.info("Démarrage du traitement en streaming")
            for batch in components['streamer'].stream_batches():
                # Nettoyage et vectorisation
                processed_texts = components['processor'].process_batch(
                    batch[config['schema']['text_column']].tolist()
                )
                X_batch = components['vectorizer'].transform(processed_texts)
                y_batch = batch[config['schema']['target_column']].values
                
                # Entraînement
                components['trainer'].partial_fit(X_batch, y_batch)
                
                # Logging périodique
                if mlflow.active_run() and (components['trainer'].n_samples_seen % config['training']['log_interval'] == 0):
                    mlflow.log_metrics({
                        'batch_size': len(batch),
                        'total_samples': components['trainer'].n_samples_seen
                    })
            
            # Finalisation et sauvegarde
            logger.info("Sauvegarde des artefacts finaux")
            components['vectorizer'].save()
            components['trainer'].save()
            
            if mlflow.active_run():
                mlflow.log_artifacts(config['paths']['artifacts_dir'])
                mlflow.log_params({
                    'total_samples': components['trainer'].n_samples_seen,
                    'model_type': config['model']['type'],
                    'text_processing': config['text_processing']['framework']['active']
                })
                
    except ConfigError as e:
        logger.error(f"Erreur de configuration: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Échec du pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("Exécution du pipeline terminée")

if __name__ == "__main__":
    run_pipeline()