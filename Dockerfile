FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du projet
COPY app/ ./app/
COPY model.pkl .

# Exposer le port
EXPOSE 8000

# Démarrer le service API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]