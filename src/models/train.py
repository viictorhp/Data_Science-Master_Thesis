"""
train.py
========
Entrena el modelo XGBoost optimizado (hiperparámetros del notebook 03)
sobre todos los datos disponibles y lo serializa con joblib.

Uso:
    python -m src.models.train

Salida:
    models/xgb_tuned.joblib   — modelo entrenado (gitignored)
    models/metadata.json      — features, parámetros, métricas CV y timestamp
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.features.preprocess import cargar_datos

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

# Hiperparámetros tuned — resultado de notebooks/03_hiperparametros.ipynb
BEST_PARAMS = {
    "subsample": 0.9,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "n_estimators": 400,
    "min_child_weight": 10,
    "max_depth": 2,
    "learning_rate": 0.01,
    "gamma": 0.5,
    "colsample_bytree": 0.5,
}


def _evaluar_cv(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"],
                         return_train_score=False)
    return {
        "acc_mean": float(res["test_accuracy"].mean()),
        "acc_std":  float(res["test_accuracy"].std()),
        "f1_mean":  float(res["test_f1_macro"].mean()),
        "f1_std":   float(res["test_f1_macro"].std()),
    }


def train():
    print("Cargando datos...")
    X, y, features = cargar_datos(modo="arbol")
    y_xgb = y - 1  # clases 0-indexed: {0=bajo, 1=medio, 2=alto}
    print(f"  {X.shape[0]} artistas | {X.shape[1]} features")
    dist = y_xgb.value_counts().sort_index().rename({0: "bajo", 1: "medio", 2: "alto"})
    print(f"  Distribución: {dist.to_dict()}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
        **BEST_PARAMS,
    )

    print("\nEvaluando con CV5 estratificado...")
    metricas = _evaluar_cv(model, X, y_xgb)
    print(f"  acc={metricas['acc_mean']:.3f} ± {metricas['acc_std']:.3f}")
    print(f"  f1_macro={metricas['f1_mean']:.3f} ± {metricas['f1_std']:.3f}")

    print("\nEntrenando sobre todos los datos...")
    model.fit(X, y_xgb)

    model_path = MODELS_DIR / "xgb_tuned.joblib"
    joblib.dump(model, model_path)
    print(f"  Guardado: {model_path}")

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(features),
        "target_encoding": {"0": "bajo", "1": "medio", "2": "alto"},
        "params": BEST_PARAMS,
        "cv5_metrics": metricas,
    }
    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"  Guardado: {meta_path}")

    return model, metadata


if __name__ == "__main__":
    train()
