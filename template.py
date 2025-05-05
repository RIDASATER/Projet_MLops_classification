import os
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "text_classification_mlops"

list_of_files = [
    # Infrastructure minimale
    ".github/workflows/main.yml",
    "Dockerfile",
    "requirements.txt",

    # Configuration
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/schema.yaml",

    # Documentation
    "README.md",
    "docs/DEVELOPMENT.md",
    "docs/DEPLOYMENT.md",

    # Code source principal
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/main.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/constants.py",

    # Data Access
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

    # API (FastAPI)
    "app/main.py",
    "app/schemas.py",
    "app/api/endpoints.py",

    # Tests (essentiels uniquement)
    "tests/test_data.py",
    "tests/test_models.py",
]

def create_project_structure():
    for filepath_str in list_of_files:
        filepath = Path(filepath_str)
        dir_path = filepath.parent

        if dir_path != Path("."):
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"✔️ Dossier créé : {dir_path}")

        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                if filepath.suffix == ".py":
                    f.write('"""\nModule docstring\n"""\n\n')
                elif filepath.name == "README.md":
                    f.write(f"# {project_name}\n\nProjet de classification de texte avec MLOps.\n")
                elif filepath.name.endswith(".yml"):
                    f.write("# Workflow GitHub Actions\n")
                elif filepath.name.endswith(".md"):
                    f.write(f"# {filepath.stem}\n\nDocumentation.\n")
            logging.info(f"📄 Fichier créé : {filepath}")
        else:
            logging.info(f"⏭️ Fichier existant : {filepath}")

     
if __name__ == "__main__":
    create_project_structure()
    logging.info("✅ Structure minimale du projet créée avec succès.")

