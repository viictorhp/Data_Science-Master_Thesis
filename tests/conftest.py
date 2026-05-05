"""
conftest.py
===========
Fixtures compartidos entre todos los módulos de test.

Los DataFrames se crean en memoria — no se necesita ningún fichero CSV
ni el modelo serializado para ejecutar la suite completa.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Lista exacta de las 31 features del modelo (modo='arbol')
# Orden extraído de construir_features() en src/models/predict.py
# ---------------------------------------------------------------------------
FEATURES_31 = [
    "sp_num_albums", "sp_num_singles", "sp_anos_activo",
    "sp_releases_por_ano", "sp_ratio_albums_singles",
    "sp_avg_duration_ms", "sp_pct_explicit", "sp_pct_colabs",
    "lfm_oyentes", "lfm_oyentes_log",
    "lfm_scrobbles", "lfm_scrobbles_log",
    "lfm_scrobbles_por_oyente", "lfm_num_generos",
    "yt_suscriptores", "yt_suscriptores_log",
    "yt_vistas_totales", "yt_vistas_log",
    "yt_num_videos", "yt_vistas_por_video", "yt_vistas_por_suscriptor",
    "sl_avg_canciones", "sl_pct_encore",
    "sl_num_paises", "sl_num_conciertos", "sl_pct_espana",
    "trend_gtrends_interes_medio", "trend_gtrends_log",
    "trend_yt_vistas_recientes", "trend_yt_vistas_recientes_log",
    "trend_yt_vistas_por_video_reciente",
]


@pytest.fixture
def features_31():
    return list(FEATURES_31)


# ---------------------------------------------------------------------------
# DataFrames mínimos para tests de build_features
# ---------------------------------------------------------------------------

@pytest.fixture
def df_spotify_discografia():
    return pd.DataFrame({
        "nombre_buscado":     ["Artista A", "Artista B"],
        "num_albums":         [2, 5],
        "num_singles":        [8, 15],
        "num_eps":            [0, 1],
        "num_total_releases": [10, 21],
        "anos_activo":        [3.0, 7.0],
        "releases_por_ano":   [3.33, 3.0],
        "primer_lanzamiento": ["2020-01-01", "2016-01-01"],
        "ultimo_lanzamiento": ["2023-06-01", "2023-01-01"],
    })


@pytest.fixture
def df_spotify_tracks():
    return pd.DataFrame({
        "nombre_artista": ["Artista A", "Artista A", "Artista B", "Artista B"],
        "artist_id":      ["id_a",      "id_a",      "id_b",      "id_b"],
        "track_id":       ["t1", "t2", "t3", "t4"],
        "track_name":     ["Track 1", "Track 2", "Track 3", "Track 4"],
        "duration_ms":    [180_000, 240_000, 210_000, 200_000],
        "explicit":       [True, False, True, True],
        "num_artistas":   [1, 2, 1, 1],
    })


@pytest.fixture
def df_lastfm():
    return pd.DataFrame({
        "nombre_buscado": ["Artista A", "Artista B"],
        "oyentes":        [50_000, 200_000],
        "scrobbles":      [1_000_000, 5_000_000],
        "generos":        ["rap,trap", ""],
        "mbid":           ["mbid_a", "mbid_b"],
        "match_score":    [0.95, 0.88],
        "artist_id":      ["id_a", "id_b"],
    })


@pytest.fixture
def df_youtube():
    return pd.DataFrame({
        "nombre_buscado": ["Artista A", "Artista B"],
        "nombre_canal":   ["Canal A", "Canal B"],
        "channel_id":     ["UC_a", "UC_b"],
        "match_score":    [0.9, 0.85],
        "suscriptores":   [10_000, 500_000],
        "vistas_totales": [1_000_000, 50_000_000],
        "num_videos":     [50, 200],
        "fecha_creacion": ["2018-01-01", "2015-06-15"],
        "pais_canal":     ["ES", "ES"],
    })


@pytest.fixture
def df_setlistfm():
    return pd.DataFrame({
        "nombre":        ["Artista A", "Artista A", "Artista B"],
        "setlistfm_mbid":["mbid_a",    "mbid_a",    "mbid_b"],
        "setlist_id":    ["sl1", "sl2", "sl3"],
        "fecha":         ["2023-05-01", "2023-07-15", "2022-12-01"],
        "venue_id":      ["v1", "v2", "v3"],
        "venue_nombre":  ["Sala 1", "Sala 2", "Sala 3"],
        "ciudad":        ["Madrid", "Barcelona", "London"],
        "pais":          ["Spain", "Spain", "United Kingdom"],
        "num_canciones": [10, 12, 15],
        "tiene_encore":  [True, False, True],
        "canciones":     ["c1|c2", "c1|c3", "c4|c5"],
        "url_setlist":   ["/s1", "/s2", "/s3"],
    })


@pytest.fixture
def df_tendencias():
    return pd.DataFrame({
        "nombre_buscado":        ["Artista A", "Artista B"],
        "gtrends_interes_medio": [35.0, 10.0],
        "yt_vistas_recientes":   [150_000, 50_000],
        "yt_videos_recientes":   [5, 5],
    })


# ---------------------------------------------------------------------------
# DataFrame mínimo con estructura de artist_features.csv (para preprocess)
# Incluye todas las columnas que cargar_datos() espera, con NaN realistas.
# ---------------------------------------------------------------------------

@pytest.fixture
def df_artist_features_minimal():
    n = 20
    rng = np.random.default_rng(42)
    niveles = ["bajo"] * 8 + ["medio"] * 7 + ["alto"] * 5

    def nan_mask(n, prob):
        return [None if rng.random() < prob else 1.0 for _ in range(n)]

    data = {
        "nombre_buscado": [f"Artista {i}" for i in range(n)],
        "nivel":          niveles,
        "tier_sala":      [1] * 8 + [2] * 7 + [3] * 5,
        "target":         [1] * 8 + [2] * 7 + [3] * 5,
        # Spotify
        "sp_num_albums":               rng.integers(0, 6, n).astype(float),
        "sp_num_singles":              rng.integers(0, 20, n).astype(float),
        "sp_num_eps":                  np.zeros(n),
        "sp_num_total_releases":       rng.integers(1, 25, n).astype(float),
        "sp_anos_activo":              rng.uniform(1, 10, n),
        "sp_releases_por_ano":         rng.uniform(0.5, 5, n),
        "sp_dias_desde_ultimo_release":rng.integers(30, 500, n).astype(float),
        "sp_ratio_albums_singles":     rng.uniform(0, 2, n),
        "sp_avg_duration_ms":          rng.uniform(160_000, 280_000, n),
        "sp_pct_explicit":             rng.uniform(0, 1, n),
        "sp_pct_colabs":               rng.uniform(0, 0.8, n),
        # Last.fm
        "lfm_oyentes":              rng.uniform(1_000, 300_000, n),
        "lfm_oyentes_log":          np.log1p(rng.uniform(1_000, 300_000, n)),
        "lfm_scrobbles":            rng.uniform(10_000, 10_000_000, n),
        "lfm_scrobbles_log":        np.log1p(rng.uniform(10_000, 10_000_000, n)),
        "lfm_scrobbles_por_oyente": rng.uniform(1, 50, n),
        "lfm_num_generos":          rng.integers(0, 6, n).astype(float),
        # YouTube
        "yt_suscriptores":          rng.uniform(1_000, 2_000_000, n),
        "yt_suscriptores_log":      np.log1p(rng.uniform(1_000, 2_000_000, n)),
        "yt_vistas_totales":        rng.uniform(100_000, 500_000_000, n),
        "yt_vistas_log":            np.log1p(rng.uniform(100_000, 500_000_000, n)),
        "yt_num_videos":            rng.integers(10, 300, n).astype(float),
        "yt_vistas_por_video":      rng.uniform(1_000, 1_000_000, n),
        "yt_vistas_por_suscriptor": rng.uniform(10, 200, n),
        "yt_edad_canal_anos":       rng.uniform(1, 10, n),
        # setlist.fm — ~40% NaN (simula artistas sin datos en setlist.fm)
        "sl_num_conciertos": [None if i % 3 == 0 else float(rng.integers(1, 50)) for i in range(n)],
        "sl_avg_canciones":  [None if i % 3 == 0 else float(rng.integers(8, 20))  for i in range(n)],
        "sl_pct_encore":     [None if i % 3 == 0 else float(rng.uniform(0, 0.5))  for i in range(n)],
        "sl_num_paises":     [None if i % 3 == 0 else float(rng.integers(1, 5))   for i in range(n)],
        "sl_pct_espana":     [None if i % 3 == 0 else float(rng.uniform(0.5, 1.0))for i in range(n)],
        # Tendencias — ~20% NaN (fallos de rate-limit)
        "trend_gtrends_interes_medio":        [None if i % 5 == 0 else float(rng.uniform(5, 80))       for i in range(n)],
        "trend_gtrends_log":                  [None if i % 5 == 0 else float(rng.uniform(1.5, 4.5))    for i in range(n)],
        "trend_yt_vistas_recientes":          [None if i % 7 == 0 else float(rng.integers(1_000, 1_000_000)) for i in range(n)],
        "trend_yt_vistas_recientes_log":      [None if i % 7 == 0 else float(rng.uniform(7, 14))       for i in range(n)],
        "trend_yt_videos_recientes":          [5.0] * n,
        "trend_yt_vistas_por_video_reciente": [None if i % 7 == 0 else float(rng.integers(1_000, 100_000)) for i in range(n)],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Mock del modelo y su metadata (para tests de predict y shap_explainer)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_metadata():
    """Copia exacta de la estructura de models/metadata.json."""
    return {
        "features": list(FEATURES_31),
        "target_encoding": {"0": "bajo", "1": "medio", "2": "alto"},
        "n_samples":  157,
        "n_features": 31,
        "trained_at": "2025-01-01T00:00:00",
        "cv5_metrics": {
            "acc_mean": 0.675, "acc_std": 0.040,
            "f1_mean":  0.669, "f1_std":  0.037,
        },
        "params": {},
    }


@pytest.fixture
def mock_model(mock_metadata):
    """
    XGBClassifier simulado que devuelve probabilidades fijas.
    predict_proba devuelve [0.2, 0.6, 0.2] → clase predicha: 'medio' (índice 1).
    """
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
    return model, mock_metadata
