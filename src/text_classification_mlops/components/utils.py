import yaml
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Charge un fichier de configuration YAML avec gestion robuste des erreurs.
    
    Args:
        config_name: Nom de la configuration (sans extension)
        
    Returns:
        Dictionnaire de configuration ou None si erreur
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        yaml.YAMLError: Si erreur de parsing
    """
    config_path = Path(f"configs/{config_name}.yaml")
    fallback_path = Path(f"configs/{config_name}_config.yaml")
    
    try:
        # Essayer les deux conventions de nommage
        if config_path.exists():
            path = config_path
        elif fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(f"Aucun fichier de configuration trouvé pour {config_name}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
            logger.info("Configuration chargée depuis %s", path)
            return config
            
    except yaml.YAMLError as e:
        logger.error("Erreur de parsing YAML dans %s: %s", path, str(e))
        raise
    except Exception as e:
        logger.error("Erreur lors du chargement de la configuration %s: %s", config_name, str(e))
        raise

def validate_text_columns(df: pd.DataFrame, required_cols: list = None) -> None:
    """
    Valide la présence des colonnes requises dans un DataFrame.
    
    Args:
        df: DataFrame à valider
        required_cols: Liste des colonnes requises (par défaut: ['text', 'sentiment'])
        
    Raises:
        ValueError: Si des colonnes requises sont manquantes
    """
    required = required_cols or ['text', 'sentiment']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        error_msg = f"Colonnes requises manquantes: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)