import os
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "text_classification_mlops"

# Structure complète du projet
list_of_files = [
    # Infrastructure
    ".github/workflows/main.yml",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    
    # Configuration
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/schema.yaml",
    
    # Documentation
    "README.md",
    "docs/DEVELOPMENT.md",
    "docs/DEPLOYMENT.md",
    
    # Source Code
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/main.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/constants.py",
    
    # Data Management
    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/database.py",
    f"src/{project_name}/data_access/local_storage.py",
    
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
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
    f"src/{project_name}/models/base_model.py",
    f"src/{project_name}/models/text_classifier.py",
    
    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/text_processing.py",
    
    # Monitoring
    f"src/{project_name}/monitoring/__init__.py",
    f"src/{project_name}/monitoring/drift_detection.py",
    
    # API
    "app/main.py",
    "app/schemas.py",
    "app/api/endpoints.py",
    
    # Tests
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_models.py",
    "tests/test_api.py",
    
    # Notebooks
    "notebooks/01_exploratory.ipynb",
    "notebooks/02_experiments.ipynb",
    
    # Data and Artifacts
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "artifacts/models/.gitkeep",
    "artifacts/vectorizers/.gitkeep",
    
    # Logs
    "logs/.gitkeep"
]

def create_project_structure():
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        # Create directory if needed
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir}")

        # Create empty file if it doesn't exist
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                if filepath.suffix == ".py":
                    f.write('"""\nModule docstring\n"""\n\n')
                elif filepath.name == "README.md":
                    f.write(f"# {project_name}\n\nProject description\n")
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")

if __name__ == "__main__":
    create_project_structure()
    logging.info("Project structure created successfully!")