import pandas as pd
import yaml
from .logger import logging

def validate_data(df: pd.DataFrame, schema_path: str) -> bool:
    """Valide que le dataset correspond au schéma attendu"""
    try:
        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        
        # Vérification des colonnes requises
        required_features = [k for k, v in schema['features'].items() if v['required']]
        missing_cols = [col for col in required_features if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        # Vérification des types
        for feature, rules in schema['features'].items():
            if feature in df.columns:
                if not pd.api.types.is_dtype_equal(df[feature].dtype, rules['type']):
                    try:
                        df[feature] = df[feature].astype(rules['type'])
                    except:
                        raise ValueError(f"Type incorrect pour {feature}. Attendu: {rules['type']}")
        
        # Vérification des valeurs autorisées pour le label
        if 'allowed_values' in schema['features']['label']:
            invalid_labels = [x for x in df['label'].unique() 
                            if x not in schema['features']['label']['allowed_values']]
            if invalid_labels:
                raise ValueError(f"Labels non autorisés trouvés: {invalid_labels}")
        
        logging.info("Validation des données réussie")
        return True
        
    except Exception as e:
        logging.error(f"Erreur de validation: {str(e)}")
        raise e