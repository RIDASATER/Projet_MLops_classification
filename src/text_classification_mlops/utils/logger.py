import logging
import os
from pathlib import Path
import sys

def setup_logging(config: dict):
    """Configure le système de logging"""
    log_dir = config['logging']['log_dir']
    log_file = config['logging']['log_file']
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )