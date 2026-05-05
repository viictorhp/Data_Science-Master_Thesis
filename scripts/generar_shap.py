"""
generar_shap.py
===============
Genera los plots SHAP globales sobre el dataset de entrenamiento y los guarda
en reports/figures/ para mostrarlos en el dashboard (página Resultados).

Requiere:
    models/xgb_tuned.joblib          (python -m src.models.train)
    data/processed/artist_features.csv

Salida:
    reports/figures/shap_summary_bar.png    — bar plot media |SHAP| global
    reports/figures/shap_beeswarm_alto.png  — beeswarm clase ALTO

Uso:
    python -m scripts.generar_shap
"""

import matplotlib
matplotlib.use("Agg")  # backend headless, sin ventana gráfica

from src.features.preprocess import cargar_datos
from src.models.shap_explainer import calcular_shap_global


def main():
    print("Cargando datos de entrenamiento (modo árbol)...")
    X, y, features = cargar_datos(modo="arbol")
    print(f"  {X.shape[0]} artistas | {X.shape[1]} features | NaN: {X.isna().sum().sum()}")

    print("\nGenerando plots SHAP globales...")
    rutas = calcular_shap_global(X)

    print("\nResumen:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")
    print("\nListo. Abre el dashboard para ver los plots en la pestaña SHAP.")


if __name__ == "__main__":
    main()
