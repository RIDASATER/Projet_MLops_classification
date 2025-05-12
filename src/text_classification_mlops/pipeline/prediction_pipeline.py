import joblib
from components.text_processing import preprocess_text

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")  # à adapter

    def predict(self, text: str):
        processed = preprocess_text(text)
        return self.model.predict([processed])[0]
