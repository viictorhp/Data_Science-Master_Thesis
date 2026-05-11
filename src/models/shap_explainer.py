"""
shap_explainer.py
=================
Explicabilidad SHAP sobre el modelo XGBoost entrenado.

Dos modos de uso:
  - Global (dataset completo): bar plot de importancia media + beeswarm para
    la clase ALTO. Se guardan como PNG en reports/figures/.
  - Individual (una predicción): waterfall plot retornado como figura matplotlib,
    listo para renderizar en Streamlit con st.pyplot().

Uso desde scripts:
    from src.models.shap_explainer import calcular_shap_global
    rutas = calcular_shap_global(X)

Uso desde Streamlit:
    from src.models.shap_explainer import shap_waterfall_fig
    fig, clase = shap_waterfall_fig(features_dict)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
"""

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"

LABEL_MAP = {0: "bajo", 1: "medio", 2: "alto"}


@lru_cache(maxsize=1)
def _cargar_explainer():
    """Carga modelo y crea TreeExplainer (cacheado en memoria)."""
    model = joblib.load(MODELS_DIR / "xgb_tuned.joblib")
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    # model se devuelve por separado: explainer.model es un TreeEnsemble interno de SHAP
    # y no expone predict_proba — necesitamos el XGBClassifier original
    return explainer, model, meta


def calcular_shap_global(
    X: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Genera y guarda los plots SHAP globales sobre el dataset de entrenamiento.

    Parámetros
    ----------
    X          : DataFrame con las features de entrenamiento (modo='arbol').
    output_dir : directorio de salida (default: reports/figures/).

    Retorna
    -------
    dict con las rutas de las dos figuras generadas:
        'summary_bar'   : bar plot de media |SHAP| (todas las clases)
        'beeswarm_alto' : beeswarm para la clase ALTO
    """
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer, _, meta = _cargar_explainer()
    features = meta["features"]
    X_model = X[features]

    print("  Calculando SHAP values sobre el dataset completo...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer(X_model)  # Explanation: (n_samples, n_features, n_classes)

    rutas = {}

    # ------------------------------------------------------------------
    # Plot 1: Bar plot — media |SHAP| global (todas las clases)
    # ------------------------------------------------------------------
    mean_abs = np.abs(sv.values).mean(axis=(0, 2))  # (n_features,)
    order    = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [features[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color="#4a90d9",
        edgecolor="white",
    )
    ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
    ax.set_title(
        "SHAP — Importancia global (media |SHAP| · todas las clases)",
        fontsize=13,
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path_bar = output_dir / "shap_summary_bar.png"
    fig.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas["summary_bar"] = path_bar
    print(f"  Guardado: {path_bar}")

    # ------------------------------------------------------------------
    # Plot 2: Beeswarm — clase ALTO (clase índice 2)
    # ------------------------------------------------------------------
    shap.plots.beeswarm(sv[:, :, 2], max_display=20, show=False)
    fig_bee = plt.gcf()
    fig_bee.set_size_inches(10, 8)
    plt.title(
        "SHAP Beeswarm — Clase ALTO (sala grande / festival)",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()
    path_bee = output_dir / "shap_beeswarm_alto.png"
    fig_bee.savefig(path_bee, dpi=150, bbox_inches="tight")
    plt.close(fig_bee)
    rutas["beeswarm_alto"] = path_bee
    print(f"  Guardado: {path_bee}")

    return rutas


def shap_waterfall_fig(
    features_dict: dict,
    max_display: int = 15,
) -> tuple[plt.Figure, str]:
    """
    Genera el waterfall plot SHAP para una predicción individual.

    Muestra las features que más empujan la predicción hacia arriba o hacia abajo
    respecto al valor base del modelo, para la clase predicha.

    Parámetros
    ----------
    features_dict : vector de features calculado por construir_features()
    max_display   : máximo de features a mostrar en el plot (default: 15)

    Retorna
    -------
    (fig, clase_nombre)
        fig          : figura matplotlib lista para st.pyplot()
        clase_nombre : "bajo" | "medio" | "alto"
    """
    explainer, model, meta = _cargar_explainer()
    features = meta["features"]

    X = pd.DataFrame([features_dict])[features]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer(X)  # Explanation: (1, n_features, n_classes)

    proba      = model.predict_proba(X)[0]
    clase_idx  = int(proba.argmax())
    clase_nombre = LABEL_MAP[clase_idx]

    # Explanation para la muestra 0 y la clase predicha
    exp = sv[0, :, clase_idx]

    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(
        f"Explicación SHAP — Clase predicha: {clase_nombre.upper()}",
        fontsize=13,
        pad=15,
    )
    plt.tight_layout()

    return fig, clase_nombre
