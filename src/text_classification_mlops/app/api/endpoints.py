from fastapi import APIRouter, Depends, HTTPException
from datetime import timedelta
from pipeline.prediction_pipeline import PredictionPipeline
from app.shemas import TextInput, PredictionOutput
from monitoring.prediction_logger import log_prediction
from .auth import create_access_token, get_current_user
from pydantic import BaseModel

router = APIRouter()

# --- Authentification ---
class UserLogin(BaseModel):
    username: str
    password: str

@router.post("/token")
def login(user: UserLogin):
    if user.username == "admin" and user.password == "adminpass":
        access_token = create_access_token(
            data={"sub": user.username, "role": "admin"},
            expires_delta=timedelta(minutes=30)
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Identifiants invalides.")

# --- Contrôle de rôle ---
def require_role(role: str):
    def checker(user_data=Depends(get_current_user)):
        if user_data.get("role") != role:
            raise HTTPException(status_code=403, detail="Accès refusé")
        return user_data
    return checker

@router.get("/admin-only")
def admin_data(current_user=Depends(require_role("admin"))):
    return {"message": "Bienvenue Admin"}

@router.get("/protected")
def protected_route(current_user=Depends(get_current_user)):
    return {"user": current_user}

@router.post("/predict", response_model=PredictionOutput)
async def predict(text_input: TextInput, current_user=Depends(get_current_user)):
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(text_input.text)
    log_prediction(input_text=text_input.text, prediction=prediction)
    return {"prediction": prediction}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
