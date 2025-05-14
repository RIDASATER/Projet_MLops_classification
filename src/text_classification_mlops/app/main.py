from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os
import time
from langdetect import detect, LangDetectException
from datetime import datetime  # Ajout pour le timestamp

app = FastAPI()

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Chargement du modèle
model = joblib.load("models/logistic_regression_model.joblib")

def get_current_timestamp():
    return datetime.now().strftime("%H:%M:%S")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "error": None,
        "text_input": "",
        "duration": None,
        "timestamp": None
    })

@app.post("/", response_class=HTMLResponse)
async def post_prediction(request: Request, text: str = Form(...)):
    error = None
    prediction = None
    duration = None
    timestamp = get_current_timestamp()
    
    try:
        # Vérification de la langue
        if len(text.strip()) < 3:
            error = "Text too short for analysis (minimum 3 characters)"
        else:
            start_detect = time.time()
            lang = detect(text)
            detect_time = time.time() - start_detect
            
            if lang != 'en':
                error = f"Please enter text in English (detected: {lang.upper()})"
            else:
                start_pred = time.time()
                prediction = model.predict([text])[0]
                pred_time = time.time() - start_pred
                
                # Temps total (détection + prédiction)
                duration = {
                    'total': round(detect_time + pred_time, 4),
                    'detection': round(detect_time, 4),
                    'prediction': round(pred_time, 4)
                }
                
    except LangDetectException:
        error = "Language detection failed - please enter English text"
    except Exception as e:
        error = f"System error: {str(e)}"
        app.logger.error(f"Error at {timestamp}: {str(e)}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "error": error,
        "duration": duration,
        "text_input": text,
        "timestamp": timestamp
    })