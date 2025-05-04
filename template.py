import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "text_classification_mlops"

list_of_files = [
    # Infrastructure
    ".github/workflows/main.yaml",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    
    # Configuration
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/schema.yaml",
    
    # Source code
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/main.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    
    # Data management
    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/database_connector.py",
    f"src/{project_name}/data_access/data_loader.py",
    
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/feature_engineering.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    
    # Pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    
    # Models
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/model.py",
    f"src/{project_name}/models/tokenizer.py",
    
    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/helpers.py",
    
    # Monitoring
    f"src/{project_name}/monitoring/__init__.py",
    f"src/{project_name}/monitoring/model_monitor.py",
    
    # API
    "app.py",
    "api/schemas.py",
    "api/routes.py",
    
    # Tests
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_model.py",
    
    # Notebooks
    "notebooks/exploratory.ipynb",
    "notebooks/experiments.ipynb",
    
    # Documentation
    "docs/getting_started.md",
    "docs/data_schema.md",
    
    # Others
    "logs/.gitkeep",
    "artifacts/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")