from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os

app = FastAPI()

# Obtenir le chemin absolu du dossier actuel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Définir les chemins vers templates et static dans le bon dossier
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Charger le modèle
model = joblib.load("models/logistic_regression_model.joblib")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request, text: str = Form(...)):
    prediction = model.predict([text])[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
