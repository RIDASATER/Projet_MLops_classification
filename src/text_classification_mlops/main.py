from components.data_ingestion import load_data
from components.data_validation import validate_data
import yaml

def main():
    # Chargement config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Pipeline
    df = load_data("configs/config.yaml")
    validate_data(df, "configs/schema.yaml")
    train, test = preprocess_data(df)
    
    # Sauvegarde
    train.to_csv(config['data']['train_path'], index=False)
    test.to_csv(config['data']['test_path'], index=False)

if __name__ == "__main__":
    main()