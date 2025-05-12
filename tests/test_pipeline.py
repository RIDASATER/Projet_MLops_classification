import pytest
import sys
from pathlib import Path

# Ajouter le chemin de 'src' au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from text_classification_mlops.components.data_ingestion import DataStreamer
from text_classification_mlops.components.text_processing import AdvancedTextCleaner
import yaml
from pathlib import Path

def test_config_files():
    """Teste la validité des fichiers de configuration"""
    for config_file in ["configs/config.yaml", 
                      "configs/training_config.yaml",
                      "configs/text_processing_config.yaml"]:
        with open(config_file) as f:
            yaml.safe_load(f)

def test_data_streamer():
    """Teste le chargement des données"""
    streamer = DataStreamer()
    batch = next(streamer.get_batches())
    assert not batch.empty
    assert 'text' in batch.columns
    assert 'sentiment' in batch.columns

def test_text_processing():
    """Teste le nettoyage de texte"""
    cleaner = AdvancedTextCleaner()
    test_cases = [
        ("Bonjour @test !", "bonjour"),
        ("http://example.com", ""),
        ("C'est un TEST.", "être test")
    ]
    for input_text, expected in test_cases:
        assert cleaner.process(input_text) == expected

def test_full_pipeline_integration():
    """Test d'intégration complet"""
    from text_classification_mlops.pipeline.training_pipeline import run_pipeline
    try:
        run_pipeline()
        assert Path("artifacts/vectorizers/tfidf_vectorizer.pkl").exists()
    except Exception as e:
        pytest.fail(f"Le pipeline a échoué: {str(e)}")