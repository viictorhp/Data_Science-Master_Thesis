"""
test_predict.py
===============
Tests de src/models/predict.py

construir_features() se testa sin modelo (puro cálculo matemático).
predecir() se testa con el modelo mockeado para no depender de
xgb_tuned.joblib (gitignoreado).
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.models.predict import construir_features, predecir, LABEL_MAP

NIVELES_VALIDOS   = {"bajo", "medio", "alto"}
N_FEATURES_MODELO = 32


# ===========================================================================
# Tests de construir_features() — puro cálculo, sin modelo
# ===========================================================================

class TestConstruirFeatures:

    # --- Estructura del vector ---

    def test_contiene_31_features(self):
        f = construir_features()
        assert len(f) == N_FEATURES_MODELO

    def test_contiene_todas_las_features_esperadas(self, features_32):
        f = construir_features()
        for feat in features_32:
            assert feat in f, f"Feature ausente: {feat}"

    def test_sin_valores_nulos_con_defaults(self):
        f = construir_features()
        for k, v in f.items():
            assert v is not None, f"None en feature {k}"
            assert not np.isnan(v), f"NaN en feature {k}"

    # --- Cálculos de logaritmos ---

    def test_log1p_oyentes(self):
        f = construir_features(lfm_oyentes=50_000)
        assert abs(f["lfm_oyentes_log"] - np.log1p(50_000)) < 1e-10

    def test_log1p_scrobbles(self):
        f = construir_features(lfm_scrobbles=1_000_000)
        assert abs(f["lfm_scrobbles_log"] - np.log1p(1_000_000)) < 1e-10

    def test_log1p_suscriptores(self):
        f = construir_features(yt_suscriptores=100_000)
        assert abs(f["yt_suscriptores_log"] - np.log1p(100_000)) < 1e-10

    def test_log1p_vistas(self):
        f = construir_features(yt_vistas_totales=5_000_000)
        assert abs(f["yt_vistas_log"] - np.log1p(5_000_000)) < 1e-10

    def test_log1p_cero_es_cero(self):
        f = construir_features(lfm_oyentes=0, yt_suscriptores=0)
        assert f["lfm_oyentes_log"] == pytest.approx(0.0)
        assert f["yt_suscriptores_log"] == pytest.approx(0.0)

    # --- Ratios de Spotify ---

    def test_releases_por_ano(self):
        # 2 albums + 8 singles = 10 releases en 4 años → 2.5/año
        f = construir_features(sp_num_albums=2, sp_num_singles=8, sp_anos_activo=4.0)
        assert abs(f["sp_releases_por_ano"] - 2.5) < 1e-6

    def test_ratio_albums_singles(self):
        # 3 albums, 9 singles → 3 / (9 + 1) = 0.3
        f = construir_features(sp_num_albums=3, sp_num_singles=9)
        assert abs(f["sp_ratio_albums_singles"] - 0.3) < 1e-6

    def test_ratio_albums_singles_sin_singles(self):
        # 5 albums, 0 singles → 5 / (0 + 1) = 5.0
        f = construir_features(sp_num_albums=5, sp_num_singles=0)
        assert abs(f["sp_ratio_albums_singles"] - 5.0) < 1e-6

    # --- Ratios de Last.fm ---

    def test_scrobbles_por_oyente(self):
        f = construir_features(lfm_oyentes=100_000, lfm_scrobbles=5_000_000)
        assert abs(f["lfm_scrobbles_por_oyente"] - 50.0) < 1e-6

    def test_scrobbles_por_oyente_sin_oyentes_no_da_nan(self):
        # Con 0 oyentes debe usar max(oyentes, 1) para evitar división por cero
        f = construir_features(lfm_oyentes=0, lfm_scrobbles=0)
        assert not np.isnan(f["lfm_scrobbles_por_oyente"])

    # --- YouTube: vistas por vídeo ---

    def test_vistas_por_video_con_num_videos(self):
        f = construir_features(yt_vistas_totales=1_000_000, yt_num_videos=100)
        assert abs(f["yt_vistas_por_video"] - 10_000.0) < 1e-6

    def test_vistas_por_video_sin_num_videos_es_cero(self):
        f = construir_features(yt_vistas_totales=5_000_000, yt_num_videos=0)
        assert f["yt_vistas_por_video"] == 0.0

    # --- Features de setlist.fm ---

    def test_sl_pct_espana_con_conciertos(self):
        # El modelo se entrenó con proporciones (0.0–1.0), no porcentajes
        f = construir_features(sl_num_conciertos=5)
        assert f["sl_pct_espana"] == 1.0

    def test_sl_pct_espana_sin_conciertos(self):
        f = construir_features(sl_num_conciertos=0)
        assert f["sl_pct_espana"] == 0.0

    def test_sl_num_paises_con_conciertos(self):
        # min(sl_num_conciertos, 1) cuando el usuario no informa países
        f = construir_features(sl_num_conciertos=10)
        assert f["sl_num_paises"] == 1.0

    def test_sl_num_paises_sin_conciertos(self):
        f = construir_features(sl_num_conciertos=0)
        assert f["sl_num_paises"] == 0.0

    # --- Tendencias (se imputan a 0 por defecto) ---

    def test_tendencias_a_cero_por_defecto(self):
        f = construir_features()
        for key in [
            "trend_gtrends_interes_medio", "trend_gtrends_log",
            "trend_yt_vistas_recientes", "trend_yt_vistas_recientes_log",
            "trend_yt_vistas_por_video_reciente",
        ]:
            assert f[key] == 0.0, f"{key} debería ser 0.0 por defecto"

    # --- Robustez ante entradas extremas ---

    def test_inputs_todos_cero_no_produce_nan(self):
        f = construir_features(
            sp_num_albums=0, sp_num_singles=0, sp_anos_activo=0.0,
            lfm_oyentes=0, lfm_scrobbles=0,
            yt_suscriptores=0, yt_vistas_totales=0,
            sl_num_conciertos=0,
        )
        for k, v in f.items():
            assert not np.isnan(v), f"NaN inesperado en {k} con inputs cero"

    def test_anos_negativos_no_produce_nan(self):
        # La función hace max(sp_anos_activo, 0.5), no debe haber NaN
        f = construir_features(sp_anos_activo=-5.0)
        for k, v in f.items():
            assert not np.isnan(v), f"NaN en {k} con años negativos"

    def test_valores_muy_grandes_no_producen_inf(self):
        f = construir_features(
            lfm_oyentes=50_000_000,
            lfm_scrobbles=1_000_000_000,
            yt_suscriptores=10_000_000,
            yt_vistas_totales=10_000_000_000,
        )
        for k, v in f.items():
            assert not np.isinf(v), f"Infinito en {k} con valores grandes"


# ===========================================================================
# Tests de predecir() con modelo mockeado
# ===========================================================================

class TestPredecir:

    def test_estructura_de_salida(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        assert "nivel" in result
        assert "probabilidades" in result
        assert "features" in result

    def test_nivel_es_valido(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        assert result["nivel"] in NIVELES_VALIDOS

    def test_probabilidades_suman_uno(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        total = sum(result["probabilidades"].values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilidades_no_negativas(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        for clase, prob in result["probabilidades"].items():
            assert prob >= 0.0, f"Probabilidad negativa para {clase}"

    def test_probabilidades_tienen_tres_clases(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        assert set(result["probabilidades"].keys()) == {"bajo", "medio", "alto"}

    def test_nivel_es_argmax_de_probabilidades(self, mock_model):
        model, meta = mock_model
        # Mock devuelve [0.2, 0.6, 0.2] → índice 1 → "medio"
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        proba = result["probabilidades"]
        clase_argmax = max(proba, key=proba.get)
        assert result["nivel"] == clase_argmax

    def test_nivel_predicho_es_medio(self, mock_model):
        """Mock devuelve [0.2, 0.6, 0.2], debería predecir 'medio'."""
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        assert result["nivel"] == "medio"

    def test_features_contiene_todas_las_claves(self, mock_model, features_32):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        for feat in features_32:
            assert feat in result["features"], f"Feature ausente en output: {feat}"

    def test_features_output_es_31(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir()
        assert len(result["features"]) == N_FEATURES_MODELO

    def test_inputs_se_reflejan_en_features_output(self, mock_model):
        model, meta = mock_model
        with patch("src.models.predict._cargar_modelo", return_value=(model, meta)):
            result = predecir(sp_num_albums=3, lfm_oyentes=80_000)
        assert result["features"]["sp_num_albums"] == 3.0
        assert result["features"]["lfm_oyentes"] == 80_000.0


# ===========================================================================
# Tests de constantes del módulo
# ===========================================================================

class TestConstantes:

    def test_label_map_cubre_tres_clases(self):
        assert len(LABEL_MAP) == 3

    def test_label_map_indices_correctos(self):
        assert LABEL_MAP[0] == "bajo"
        assert LABEL_MAP[1] == "medio"
        assert LABEL_MAP[2] == "alto"

    def test_label_map_valores_son_niveles_validos(self):
        assert set(LABEL_MAP.values()) == NIVELES_VALIDOS
