"""
predict.py
==========
Módulo de inferencia. Carga el modelo serializado y expone predecir()
para obtener el tier de sala de un artista a partir de sus métricas básicas.

El usuario solo proporciona los campos accesibles. El resto de features
se computan automáticamente (logs, ratios) o se imputan con valores
neutros (0 para tendencias, medianas para campos secundarios).

Uso desde Python:
    from src.models.predict import predecir
    resultado = predecir(sp_num_albums=2, sp_num_singles=8, ...)
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

LABEL_MAP = {0: "bajo", 1: "medio", 2: "alto"}

# Medianas del dataset de entrenamiento para campos secundarios no pedidos al usuario.
# Permiten una imputación más realista que el 0 para artistas con algo de presencia.
_DEFAULTS = {
    "sp_avg_duration_ms": 210_000,  # ~3:30 min, típico rap/urbano español
    "sp_pct_explicit":    0.6,       # mediana dataset — mayoría de rap es explicit
    "sp_pct_colabs":      0.3,       # mediana dataset
    "lfm_num_generos":    3,         # mediana dataset
    "yt_num_videos":      0,         # se pide 0 — no disponible sin API
}


@lru_cache(maxsize=1)
def _cargar_modelo():
    model = joblib.load(MODELS_DIR / "xgb_tuned.joblib")
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def _log1p(x: float) -> float:
    return float(np.log1p(max(x, 0)))


def construir_features(
    sp_num_albums: int = 0,
    sp_num_singles: int = 0,
    sp_anos_activo: float = 1.0,
    lfm_oyentes: int = 0,
    lfm_scrobbles: int = 0,
    yt_suscriptores: int = 0,
    yt_vistas_totales: int = 0,
    sl_num_conciertos: int = 0,
) -> dict:
    """
    Construye el vector completo de 31 features a partir de los inputs del usuario.
    Todos los campos derivados (logs, ratios) se calculan aquí.
    Las features no pedidas al usuario se imputan con valores neutros.
    """
    anos = max(sp_anos_activo, 0.5)
    total_releases = sp_num_albums + sp_num_singles

    # YouTube: si no hay suscriptores ni vistas, yt_vistas_por_video = 0
    yt_num_videos = _DEFAULTS["yt_num_videos"]
    yt_vistas_por_video = (
        yt_vistas_totales / yt_num_videos if yt_num_videos > 0 else 0.0
    )

    return {
        # --- Spotify ---
        "sp_num_albums":            float(sp_num_albums),
        "sp_num_singles":           float(sp_num_singles),
        "sp_anos_activo":           float(sp_anos_activo),
        "sp_releases_por_ano":      total_releases / anos,
        "sp_ratio_albums_singles":  sp_num_albums / (sp_num_singles + 1),
        "sp_avg_duration_ms":       _DEFAULTS["sp_avg_duration_ms"],
        "sp_pct_explicit":          _DEFAULTS["sp_pct_explicit"],
        "sp_pct_colabs":            _DEFAULTS["sp_pct_colabs"],
        # --- Last.fm ---
        "lfm_oyentes":              float(lfm_oyentes),
        "lfm_oyentes_log":          _log1p(lfm_oyentes),
        "lfm_scrobbles":            float(lfm_scrobbles),
        "lfm_scrobbles_log":        _log1p(lfm_scrobbles),
        "lfm_scrobbles_por_oyente": lfm_scrobbles / max(lfm_oyentes, 1),
        "lfm_num_generos":          _DEFAULTS["lfm_num_generos"],
        # --- YouTube ---
        "yt_suscriptores":          float(yt_suscriptores),
        "yt_suscriptores_log":      _log1p(yt_suscriptores),
        "yt_vistas_totales":        float(yt_vistas_totales),
        "yt_vistas_log":            _log1p(yt_vistas_totales),
        "yt_num_videos":            float(yt_num_videos),
        "yt_vistas_por_video":      float(yt_vistas_por_video),
        "yt_vistas_por_suscriptor": yt_vistas_totales / max(yt_suscriptores, 1),
        # --- setlist.fm ---
        "sl_avg_canciones":         0.0,
        "sl_pct_encore":            0.0,
        "sl_num_paises":            float(min(sl_num_conciertos, 1)),
        "sl_num_conciertos":        float(sl_num_conciertos),
        "sl_pct_espana":            100.0 if sl_num_conciertos > 0 else 0.0,
        # --- Tendencias (no se piden, se imputan a 0) ---
        "trend_gtrends_interes_medio":        0.0,
        "trend_gtrends_log":                  0.0,
        "trend_yt_vistas_recientes":          0.0,
        "trend_yt_vistas_recientes_log":      0.0,
        "trend_yt_vistas_por_video_reciente": 0.0,
    }


def predecir(
    sp_num_albums: int = 0,
    sp_num_singles: int = 0,
    sp_anos_activo: float = 1.0,
    lfm_oyentes: int = 0,
    lfm_scrobbles: int = 0,
    yt_suscriptores: int = 0,
    yt_vistas_totales: int = 0,
    sl_num_conciertos: int = 0,
) -> dict:
    """
    Predice el tier de sala de un artista de rap/urbano español.

    Parámetros
    ----------
    sp_num_albums       : Nº de álbumes en Spotify
    sp_num_singles      : Nº de singles en Spotify
    sp_anos_activo      : Años desde el primer lanzamiento
    lfm_oyentes         : Oyentes únicos históricos en Last.fm
    lfm_scrobbles       : Reproducciones totales acumuladas en Last.fm
    yt_suscriptores     : Suscriptores del canal de YouTube
    yt_vistas_totales   : Vistas totales acumuladas en YouTube
    sl_num_conciertos   : Nº de conciertos documentados en setlist.fm

    Retorna
    -------
    dict con:
        nivel           : "bajo" | "medio" | "alto"
        probabilidades  : {"bajo": float, "medio": float, "alto": float}
        features        : vector completo de 31 features enviado al modelo
    """
    model, meta = _cargar_modelo()

    features_dict = construir_features(
        sp_num_albums=sp_num_albums,
        sp_num_singles=sp_num_singles,
        sp_anos_activo=sp_anos_activo,
        lfm_oyentes=lfm_oyentes,
        lfm_scrobbles=lfm_scrobbles,
        yt_suscriptores=yt_suscriptores,
        yt_vistas_totales=yt_vistas_totales,
        sl_num_conciertos=sl_num_conciertos,
    )

    # Construir DataFrame respetando el orden exacto que espera el modelo
    X = pd.DataFrame([features_dict])[meta["features"]]
    proba = model.predict_proba(X)[0]  # [p_bajo, p_medio, p_alto]

    nivel_idx = int(proba.argmax())

    return {
        "nivel":          LABEL_MAP[nivel_idx],
        "probabilidades": {
            "bajo":  round(float(proba[0]), 4),
            "medio": round(float(proba[1]), 4),
            "alto":  round(float(proba[2]), 4),
        },
        "features": features_dict,
    }


if __name__ == "__main__":
    # Prueba rápida — artista tipo emergente sin conciertos
    resultado = predecir(
        sp_num_albums=1,
        sp_num_singles=6,
        sp_anos_activo=2,
        lfm_oyentes=9_000,
        lfm_scrobbles=300_000,
        yt_suscriptores=6_000,
        yt_vistas_totales=800_000,
        sl_num_conciertos=0,
    )
    print(f"Nivel predicho : {resultado['nivel']}")
    print("Probabilidades :")
    for k, v in resultado["probabilidades"].items():
        print(f"  {k:5s}: {v:.1%}")
