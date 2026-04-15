"""
Script d'entraînement - Dataset Iris
Auteur : Coulibaly Mohamed Abdulaziz
"""

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train():
    print("=== Chargement du dataset Iris ===")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    print(f"Nombre d'échantillons : {len(X)}")
    print(f"Classes : {list(iris.target_names)}")

    print("\n=== Prétraitement ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    print("\n=== Entraînement du modèle ===")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    print("\n=== Évaluation ===")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\n=== Sauvegarde du modèle ===")
    model_data = {
        "pipeline": pipeline,
        "feature_names": iris.feature_names,
        "target_names": list(iris.target_names)
    }
    joblib.dump(model_data, "model.pkl")
    print("Modèle sauvegardé : model.pkl")
    print(f"\nRésultat final => Accuracy : {acc:.4f}")

if __name__ == "__main__":
    train()