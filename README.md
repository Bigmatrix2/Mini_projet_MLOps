# Mini Projet MLOps - Iris Prediction API

**Auteur :** Coulibaly Mohamed Abdulaziz  
**Dépôt :** https://github.com/Bigmatrix2/Mini_projet_MLOps

## Dataset
Iris Dataset (sklearn) — classification de fleurs en 3 classes.

## Stack
- Python 3.11 · scikit-learn · FastAPI · Docker · GitHub Actions

## Structure
```
Mini_projet_MLOps/
├── train.py                    # Script d'entraînement
├── app/
│   └── main.py                 # API FastAPI
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml    # Pipeline CI
```

## Lancer localement

```bash
# 1. Entraîner le modèle
pip install -r requirements.txt
python train.py

# 2. Démarrer l'API
uvicorn app.main:app --reload

# 3. Test health
curl http://localhost:8000/health

# 4. Test prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Docker

```bash
python train.py          # générer model.pkl
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## Pipeline CI

| Branche      | Jobs exécutés                              |
|--------------|---------------------------------------------|
| `feature/**` | Install deps → Train model                  |
| `develop`    | Install deps → Train model → Build → Push   |