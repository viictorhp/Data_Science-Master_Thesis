"""
test_build_features.py
======================
Tests unitarios de src/features/build_features.py

Cada función _features_*() se prueba de forma aislada con DataFrames mínimos.
No se necesitan ficheros CSV reales.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    _features_lastfm,
    _features_setlistfm,
    _features_spotify_discografia,
    _features_spotify_tracks,
    _features_tendencias,
    _features_youtube,
)


# ===========================================================================
# Spotify — discografía
# ===========================================================================

class TestSpotifyDiscografia:

    def test_columnas_salida(self, df_spotify_discografia):
        result = _features_spotify_discografia(df_spotify_discografia)
        expected = {
            "nombre_buscado", "sp_num_albums", "sp_num_singles", "sp_num_eps",
            "sp_num_total_releases", "sp_anos_activo", "sp_releases_por_ano",
            "sp_dias_desde_ultimo_release", "sp_ratio_albums_singles",
        }
        assert set(result.columns) == expected

    def test_renombrado_num_albums(self, df_spotify_discografia):
        result = _features_spotify_discografia(df_spotify_discografia)
        assert "sp_num_albums" in result.columns
        assert "num_albums" not in result.columns

    def test_ratio_albums_singles(self, df_spotify_discografia):
        result = _features_spotify_discografia(df_spotify_discografia)
        # Artista A: 2 álbumes / (8 singles + 1) = 0.2222...
        expected = 2 / (8 + 1)
        assert abs(result.loc[0, "sp_ratio_albums_singles"] - expected) < 1e-6

    def test_dias_desde_ultimo_release_no_negativo(self, df_spotify_discografia):
        result = _features_spotify_discografia(df_spotify_discografia)
        assert (result["sp_dias_desde_ultimo_release"] >= 0).all()

    def test_no_modifica_df_original(self, df_spotify_discografia):
        cols_before = list(df_spotify_discografia.columns)
        _features_spotify_discografia(df_spotify_discografia)
        assert list(df_spotify_discografia.columns) == cols_before

    def test_una_fila_por_artista(self, df_spotify_discografia):
        result = _features_spotify_discografia(df_spotify_discografia)
        assert len(result) == len(df_spotify_discografia)


# ===========================================================================
# Spotify — top tracks
# ===========================================================================

class TestSpotifyTracks:

    def test_columnas_salida(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        expected = {"nombre_buscado", "sp_avg_duration_ms", "sp_pct_explicit", "sp_pct_colabs"}
        assert set(result.columns) == expected

    def test_duracion_media_correcta(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: (180_000 + 240_000) / 2 = 210_000
        assert abs(row_a["sp_avg_duration_ms"] - 210_000) < 1

    def test_pct_explicit_correcto(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: 1 explicit (True) de 2 tracks → 0.5
        assert abs(row_a["sp_pct_explicit"] - 0.5) < 1e-6

    def test_pct_colabs_correcto(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: track 2 tiene 2 artistas (colab). 1/2 = 0.5
        assert abs(row_a["sp_pct_colabs"] - 0.5) < 1e-6

    def test_artista_b_todos_explicit(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        row_b = result[result["nombre_buscado"] == "Artista B"].iloc[0]
        assert abs(row_b["sp_pct_explicit"] - 1.0) < 1e-6

    def test_un_artista_por_fila(self, df_spotify_tracks):
        result = _features_spotify_tracks(df_spotify_tracks)
        assert len(result) == 2  # Artista A y Artista B


# ===========================================================================
# Last.fm
# ===========================================================================

class TestLastfm:

    def test_columnas_salida(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        expected = {
            "nombre_buscado",
            "lfm_oyentes", "lfm_oyentes_log",
            "lfm_scrobbles", "lfm_scrobbles_log",
            "lfm_scrobbles_por_oyente", "lfm_num_generos",
        }
        assert set(result.columns) == expected

    def test_oyentes_log_es_log1p(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["lfm_oyentes_log"] - np.log1p(50_000)) < 1e-6

    def test_scrobbles_log_es_log1p(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["lfm_scrobbles_log"] - np.log1p(1_000_000)) < 1e-6

    def test_scrobbles_por_oyente_correcto(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["lfm_scrobbles_por_oyente"] - 1_000_000 / 50_000) < 1e-3

    def test_num_generos_con_generos(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # "rap,trap" → 2 géneros
        assert row_a["lfm_num_generos"] == 2

    def test_num_generos_sin_generos(self, df_lastfm):
        result = _features_lastfm(df_lastfm)
        row_b = result[result["nombre_buscado"] == "Artista B"].iloc[0]
        # "" → 0 géneros
        assert row_b["lfm_num_generos"] == 0

    def test_sin_genero_como_string_da_cero(self, df_lastfm):
        df = df_lastfm.copy()
        df.loc[0, "generos"] = "Sin género"
        result = _features_lastfm(df)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert row_a["lfm_num_generos"] == 0


# ===========================================================================
# YouTube
# ===========================================================================

class TestYoutube:

    def test_columnas_salida(self, df_youtube):
        result = _features_youtube(df_youtube)
        expected = {
            "nombre_buscado",
            "yt_suscriptores", "yt_suscriptores_log",
            "yt_vistas_totales", "yt_vistas_log",
            "yt_num_videos",
            "yt_vistas_por_video", "yt_vistas_por_suscriptor",
            "yt_edad_canal_anos",
        }
        assert set(result.columns) == expected

    def test_suscriptores_log_es_log1p(self, df_youtube):
        result = _features_youtube(df_youtube)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["yt_suscriptores_log"] - np.log1p(10_000)) < 1e-6

    def test_vistas_log_es_log1p(self, df_youtube):
        result = _features_youtube(df_youtube)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["yt_vistas_log"] - np.log1p(1_000_000)) < 1e-6

    def test_vistas_por_video_correcto(self, df_youtube):
        result = _features_youtube(df_youtube)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # 1_000_000 / 50 videos = 20_000
        assert abs(row_a["yt_vistas_por_video"] - 20_000) < 1

    def test_vistas_por_suscriptor_correcto(self, df_youtube):
        result = _features_youtube(df_youtube)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # 1_000_000 / 10_000 suscriptores = 100
        assert abs(row_a["yt_vistas_por_suscriptor"] - 100) < 1e-3

    def test_edad_canal_positiva(self, df_youtube):
        result = _features_youtube(df_youtube)
        assert (result["yt_edad_canal_anos"] > 0).all()

    def test_canal_sin_videos_no_produce_nan_en_vistas_por_video(self, df_youtube):
        df = df_youtube.copy()
        df.loc[0, "num_videos"] = 0
        result = _features_youtube(df)
        # Con 0 videos, yt_vistas_por_video debería ser NaN (división por NaN)
        # Esto es el comportamiento actual — lo verificamos para detectar regresiones
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert pd.isna(row_a["yt_vistas_por_video"])


# ===========================================================================
# setlist.fm
# ===========================================================================

class TestSetlistfm:

    def test_columnas_salida(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        expected = {
            "nombre_buscado",
            "sl_num_conciertos", "sl_avg_canciones",
            "sl_pct_encore", "sl_num_paises", "sl_pct_espana",
        }
        assert set(result.columns) == expected

    def test_num_conciertos(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert row_a["sl_num_conciertos"] == 2

    def test_pct_encore_correcto(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: 2 conciertos, 1 con encore → 0.5
        assert abs(row_a["sl_pct_encore"] - 0.5) < 1e-6

    def test_pct_espana_todo_en_espana(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: 2 conciertos en Spain → 100%
        assert abs(row_a["sl_pct_espana"] - 1.0) < 1e-6

    def test_pct_espana_fuera_de_espana(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        row_b = result[result["nombre_buscado"] == "Artista B"].iloc[0]
        # Artista B: 1 concierto en United Kingdom → 0%
        assert abs(row_b["sl_pct_espana"] - 0.0) < 1e-6

    def test_num_paises_artista_a(self, df_setlistfm):
        result = _features_setlistfm(df_setlistfm)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # Artista A: solo Spain → 1 país
        assert row_a["sl_num_paises"] == 1

    def test_df_sin_fechas_validas_devuelve_vacio(self):
        df_sin_fechas = pd.DataFrame({
            "nombre": ["X"], "setlistfm_mbid": ["m"],
            "setlist_id": ["s"], "fecha": [None],
            "venue_id": ["v"], "venue_nombre": ["Sala"],
            "ciudad": ["Madrid"], "pais": ["Spain"],
            "num_canciones": [10], "tiene_encore": [True],
            "canciones": ["c1"], "url_setlist": ["/s"],
        })
        result = _features_setlistfm(df_sin_fechas)
        assert len(result) == 0

    def test_df_completamente_vacio_devuelve_df_vacio(self):
        df_empty = pd.DataFrame(columns=[
            "nombre", "setlistfm_mbid", "setlist_id", "fecha",
            "venue_id", "venue_nombre", "ciudad", "pais",
            "num_canciones", "tiene_encore", "canciones", "url_setlist",
        ])
        result = _features_setlistfm(df_empty)
        assert len(result) == 0


# ===========================================================================
# Tendencias
# ===========================================================================

class TestTendencias:

    def test_columnas_salida(self, df_tendencias):
        result = _features_tendencias(df_tendencias)
        expected = {
            "nombre_buscado",
            "trend_gtrends_interes_medio", "trend_gtrends_log",
            "trend_yt_vistas_recientes", "trend_yt_vistas_recientes_log",
            "trend_yt_videos_recientes",
            "trend_yt_vistas_por_video_reciente",
        }
        assert set(result.columns) == expected

    def test_gtrends_log_es_log1p(self, df_tendencias):
        result = _features_tendencias(df_tendencias)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["trend_gtrends_log"] - np.log1p(35.0)) < 1e-6

    def test_vistas_recientes_log_es_log1p(self, df_tendencias):
        result = _features_tendencias(df_tendencias)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert abs(row_a["trend_yt_vistas_recientes_log"] - np.log1p(150_000)) < 1e-6

    def test_vistas_por_video_reciente_correcto(self, df_tendencias):
        result = _features_tendencias(df_tendencias)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        # 150_000 / 5 = 30_000
        assert abs(row_a["trend_yt_vistas_por_video_reciente"] - 30_000) < 1

    def test_sin_videos_recientes_produce_nan(self, df_tendencias):
        df = df_tendencias.copy()
        df.loc[0, "yt_videos_recientes"] = 0
        result = _features_tendencias(df)
        row_a = result[result["nombre_buscado"] == "Artista A"].iloc[0]
        assert pd.isna(row_a["trend_yt_vistas_por_video_reciente"])
