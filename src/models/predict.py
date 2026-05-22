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
from typing import Optional

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
    # --- Parámetros originales (sin cambio) ---
    sp_num_albums: int = 0,
    sp_num_singles: int = 0,
    sp_anos_activo: float = 1.0,
    lfm_oyentes: int = 0,
    lfm_scrobbles: int = 0,
    yt_suscriptores: int = 0,
    yt_vistas_totales: int = 0,
    yt_num_videos: int = 0,
    sl_num_conciertos: int = 0,
    sl_tiene_datos: int = 0,
    # --- Parámetros nuevos con defaults (retrocompatibles) ---
    # Antes eran constantes en _DEFAULTS; ahora el fetch automático provee valores reales
    sp_avg_duration_ms: float = _DEFAULTS["sp_avg_duration_ms"],
    sp_pct_explicit: float = _DEFAULTS["sp_pct_explicit"],
    sp_pct_colabs: float = _DEFAULTS["sp_pct_colabs"],
    lfm_num_generos: int = _DEFAULTS["lfm_num_generos"],
    # Setlist.fm: antes hardcodeados a 0 o inferidos incorrectamente
    sl_avg_canciones: float = 0.0,
    sl_pct_encore: float = 0.0,
    sl_num_paises: Optional[int] = None,    # None = inferir de sl_num_conciertos
    sl_pct_espana: Optional[float] = None,  # None = 100% si hay conciertos, 0% si no
    # Tendencias: antes siempre 0; ahora reciben el valor real del fetch
    trend_gtrends_interes_medio: float = 0.0,
    trend_yt_vistas_recientes: float = 0.0,
    trend_yt_videos_recientes: int = 0,
) -> dict:
    """
    Construye el vector completo de 32 features.
    Los parámetros nuevos tienen defaults que replican el comportamiento anterior,
    por lo que el formulario manual y los tests existentes siguen funcionando.
    """
    anos = max(sp_anos_activo, 0.5)
    total_releases = sp_num_albums + sp_num_singles
    sl_tiene_datos_final = 1 if sl_num_conciertos > 0 else int(sl_tiene_datos)

    yt_vistas_por_video = (
        yt_vistas_totales / yt_num_videos if yt_num_videos > 0 else 0.0
    )

    # Fallbacks para campos setlist.fm opcionales
    sl_num_paises_final = (
        sl_num_paises if sl_num_paises is not None
        else min(sl_num_conciertos, 1)
    )
    sl_pct_espana_final = (
        sl_pct_espana if sl_pct_espana is not None
        else (1.0 if sl_num_conciertos > 0 else 0.0)
    )

    # Tendencias derivadas
    trend_gtrends_log = _log1p(trend_gtrends_interes_medio)
    trend_yt_vistas_recientes_log = _log1p(trend_yt_vistas_recientes)
    trend_yt_vistas_por_video_reciente = (
        trend_yt_vistas_recientes / trend_yt_videos_recientes
        if trend_yt_videos_recientes > 0 else 0.0
    )

    return {
        # --- Spotify ---
        "sp_num_albums":            float(sp_num_albums),
        "sp_num_singles":           float(sp_num_singles),
        "sp_anos_activo":           float(sp_anos_activo),
        "sp_releases_por_ano":      total_releases / anos,
        "sp_ratio_albums_singles":  sp_num_albums / (sp_num_singles + 1),
        "sp_avg_duration_ms":       float(sp_avg_duration_ms),
        "sp_pct_explicit":          float(sp_pct_explicit),
        "sp_pct_colabs":            float(sp_pct_colabs),
        # --- Last.fm ---
        "lfm_oyentes":              float(lfm_oyentes),
        "lfm_oyentes_log":          _log1p(lfm_oyentes),
        "lfm_scrobbles":            float(lfm_scrobbles),
        "lfm_scrobbles_log":        _log1p(lfm_scrobbles),
        "lfm_scrobbles_por_oyente": lfm_scrobbles / max(lfm_oyentes, 1),
        "lfm_num_generos":          float(lfm_num_generos),
        # --- YouTube ---
        "yt_suscriptores":          float(yt_suscriptores),
        "yt_suscriptores_log":      _log1p(yt_suscriptores),
        "yt_vistas_totales":        float(yt_vistas_totales),
        "yt_vistas_log":            _log1p(yt_vistas_totales),
        "yt_num_videos":            float(yt_num_videos),
        "yt_vistas_por_video":      float(yt_vistas_por_video),
        "yt_vistas_por_suscriptor": yt_vistas_totales / max(yt_suscriptores, 1),
        # --- setlist.fm ---
        "sl_avg_canciones":         float(sl_avg_canciones),
        "sl_pct_encore":            float(sl_pct_encore),
        "sl_num_paises":            float(sl_num_paises_final),
        "sl_num_conciertos":        float(sl_num_conciertos),
        "sl_pct_espana":            float(sl_pct_espana_final),
        "sl_tiene_datos":           float(sl_tiene_datos_final),
        # --- Tendencias ---
        "trend_gtrends_interes_medio":        float(trend_gtrends_interes_medio),
        "trend_gtrends_log":                  trend_gtrends_log,
        "trend_yt_vistas_recientes":          float(trend_yt_vistas_recientes),
        "trend_yt_vistas_recientes_log":      trend_yt_vistas_recientes_log,
        "trend_yt_vistas_por_video_reciente": trend_yt_vistas_por_video_reciente,
    }


def predecir(
    sp_num_albums: int = 0,
    sp_num_singles: int = 0,
    sp_anos_activo: float = 1.0,
    lfm_oyentes: int = 0,
    lfm_scrobbles: int = 0,
    yt_suscriptores: int = 0,
    yt_vistas_totales: int = 0,
    yt_num_videos: int = 0,
    sl_num_conciertos: int = 0,
    sl_tiene_datos: int = 0,
    # Parámetros nuevos (retrocompatibles: todos tienen defaults)
    sp_avg_duration_ms: float = _DEFAULTS["sp_avg_duration_ms"],
    sp_pct_explicit: float = _DEFAULTS["sp_pct_explicit"],
    sp_pct_colabs: float = _DEFAULTS["sp_pct_colabs"],
    lfm_num_generos: int = _DEFAULTS["lfm_num_generos"],
    sl_avg_canciones: float = 0.0,
    sl_pct_encore: float = 0.0,
    sl_num_paises: Optional[int] = None,
    sl_pct_espana: Optional[float] = None,
    trend_gtrends_interes_medio: float = 0.0,
    trend_yt_vistas_recientes: float = 0.0,
    trend_yt_videos_recientes: int = 0,
) -> dict:
    """
    Predice el tier de sala de un artista de rap/urbano español.
    Los parámetros nuevos tienen defaults que mantienen el comportamiento anterior.

    Retorna dict con: nivel, probabilidades, features.
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
        yt_num_videos=yt_num_videos,
        sl_num_conciertos=sl_num_conciertos,
        sl_tiene_datos=sl_tiene_datos,
        sp_avg_duration_ms=sp_avg_duration_ms,
        sp_pct_explicit=sp_pct_explicit,
        sp_pct_colabs=sp_pct_colabs,
        lfm_num_generos=lfm_num_generos,
        sl_avg_canciones=sl_avg_canciones,
        sl_pct_encore=sl_pct_encore,
        sl_num_paises=sl_num_paises,
        sl_pct_espana=sl_pct_espana,
        trend_gtrends_interes_medio=trend_gtrends_interes_medio,
        trend_yt_vistas_recientes=trend_yt_vistas_recientes,
        trend_yt_videos_recientes=trend_yt_videos_recientes,
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
