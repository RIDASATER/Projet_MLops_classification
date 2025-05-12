import logging
from datetime import datetime, timezone

logging.basicConfig(filename="predictions.log", level=logging.INFO)

def log_prediction(input_text: str, prediction):
    now = datetime.now(timezone.utc)
    logging.info(f"{now.isoformat()} | Input: {input_text} | Prediction: {prediction}")
