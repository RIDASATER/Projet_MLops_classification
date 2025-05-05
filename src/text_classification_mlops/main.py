import os
import sys
import yaml

# Ajout du chemin absolu du projet
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.text_classification_mlops.components.data_ingestion import load_dataset
from src.text_classification_mlopscomponents.data_validation import validate_data
from src.text_classification_mlops.logger import setup_logging

def main():
    # Chargement de la configuration
    config_path = os.path.join(project_root, "configs/config.yaml")
    schema_path = os.path.join(project_root, "configs/schema.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Configuration du logging
    setup_logging(config)
    
    try:
        # 1. Chargement des données
        df = load_dataset(config_path)
        
        # 2. Validation des données
        is_valid = validate_data(df, schema_path)
        
        if is_valid:
            print("Données validées avec succès. Prêtes pour l'analyse!")
            print("\nAperçu des données:")
            print(df.head())
            print(f"\nNombre d'échantillons: {len(df)}")
            print(f"Distribution des classes:\n{df['label'].value_counts()}")
            
    except Exception as e:
        print(f"Erreur dans le pipeline: {str(e)}")

if __name__ == "__main__":
    main()