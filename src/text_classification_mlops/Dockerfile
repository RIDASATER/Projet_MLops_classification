# Étape 1 : image de base
FROM python:3.11-slim

# Étape 2 : définir le répertoire de travail
WORKDIR /app

# Étape 3 : copier tous les fichiers dans l'image
COPY . .

# Étape 4 : installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Étape 5 : exposer le port 8000 pour FastAPI
EXPOSE 8000

# Étape 6 : lancer le serveur FastAPI avec Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

