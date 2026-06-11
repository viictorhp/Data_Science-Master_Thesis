"""
test_preprocess.py
==================
Tests unitarios de src/features/preprocess.py

La función _imputar() se testa con DataFrames en memoria.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocess import (
    FEATURES_ELIMINAR,
    FEATURES_RAW_SOLO_ARBOL,
    _imputar,
    cargar_datos,
)


# ===========================================================================
# Tests de _imputar()
# ===========================================================================

class TestImputar:

    def test_sl_conteos_nan_a_cero(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        assert result["sl_num_conciertos"].isna().sum() == 0
        assert result["sl_num_paises"].isna().sum() == 0
        # Los que eran NaN deben valer exactamente 0
        was_nan = df_artist_features_minimal["sl_num_conciertos"].isna()
        assert (result.loc[was_nan, "sl_num_conciertos"] == 0).all()

    def test_sl_num_paises_nan_a_cero(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        was_nan = df_artist_features_minimal["sl_num_paises"].isna()
        assert (result.loc[was_nan, "sl_num_paises"] == 0).all()

    def test_sl_ratios_nan_a_mediana(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        for col in ["sl_avg_canciones", "sl_pct_encore", "sl_pct_espana"]:
            assert result[col].isna().sum() == 0, f"{col} sigue teniendo NaN"

    def test_yt_vistas_nan_a_mediana_por_nivel(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        for col in ["yt_vistas_totales", "yt_vistas_log", "yt_num_videos",
                    "yt_vistas_por_video", "yt_vistas_por_suscriptor"]:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"{col} sigue teniendo NaN"

    def test_gtrends_nan_a_mediana_por_nivel(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        assert result["trend_gtrends_interes_medio"].isna().sum() == 0
        assert result["trend_gtrends_log"].isna().sum() == 0

    def test_sin_nan_tras_imputacion_completa(self, df_artist_features_minimal):
        result = _imputar(df_artist_features_minimal)
        assert result.isna().sum().sum() == 0

    def test_no_modifica_df_original(self, df_artist_features_minimal):
        nan_count_original = df_artist_features_minimal.isna().sum().sum()
        _imputar(df_artist_features_minimal)
        assert df_artist_features_minimal.isna().sum().sum() == nan_count_original

    def test_valores_no_nan_no_cambian(self, df_artist_features_minimal):
        df = df_artist_features_minimal.copy()
        # Guardamos un valor conocido sin NaN
        val_original = df.loc[1, "sp_num_albums"]
        result = _imputar(df)
        assert result.loc[1, "sp_num_albums"] == val_original

    def test_df_sin_nan_pasa_sin_cambios_numericos(self):
        """Si no hay NaN, _imputar no debe cambiar los valores numéricos."""
        df = pd.DataFrame({
            "nombre_buscado": ["X"],
            "nivel": ["bajo"],
            "sl_num_conciertos": [5.0],
            "sl_num_paises": [2.0],
            "sl_avg_canciones": [12.0],
            "sl_pct_encore": [0.3],
            "sl_pct_espana": [0.8],
            "yt_vistas_totales": [1_000_000.0],
            "yt_vistas_log": [13.8],
            "yt_num_videos": [50.0],
            "yt_vistas_por_video": [20_000.0],
            "yt_vistas_por_suscriptor": [100.0],
            "trend_gtrends_interes_medio": [30.0],
            "trend_gtrends_log": [3.4],
            "trend_yt_vistas_recientes": [50_000.0],
            "trend_yt_vistas_recientes_log": [10.8],
            "trend_yt_videos_recientes": [5.0],
            "trend_yt_vistas_por_video_reciente": [10_000.0],
        })
        result = _imputar(df)
        assert result.loc[0, "sl_num_conciertos"] == 5.0
        assert result.loc[0, "sl_pct_encore"] == pytest.approx(0.3)


# ===========================================================================
# Tests de cargar_datos() con CSV temporal
# ===========================================================================

class TestCargarDatos:

    def test_modo_invalido_lanza_valueerror(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="modo debe ser"):
            cargar_datos(modo="invalido", csv_path=csv)

    def test_modo_arbol_no_tiene_nan(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        X, y, _ = cargar_datos(modo="arbol", csv_path=csv)
        assert X.isna().sum().sum() == 0

    def test_modo_lineal_no_tiene_nan(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        X, y, _ = cargar_datos(modo="lineal", csv_path=csv)
        assert X.isna().sum().sum() == 0

    def test_arbol_mas_features_que_lineal(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, _, feats_arbol  = cargar_datos(modo="arbol",  csv_path=csv)
        _, _, feats_lineal = cargar_datos(modo="lineal", csv_path=csv)
        assert len(feats_arbol) > len(feats_lineal)

    def test_target_solo_valores_validos(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, y, _ = cargar_datos(modo="arbol", csv_path=csv)
        assert set(y.unique()).issubset({1, 2, 3})

    def test_features_eliminadas_no_estan_en_arbol(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, _, features = cargar_datos(modo="arbol", csv_path=csv)
        for feat in FEATURES_ELIMINAR:
            assert feat not in features, f"{feat} debería estar excluida (arbol)"

    def test_columnas_meta_no_estan_en_features(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, _, features = cargar_datos(modo="arbol", csv_path=csv)
        for meta_col in ["nombre_buscado", "nivel", "tier_sala", "target"]:
            assert meta_col not in features

    def test_raw_features_no_estan_en_lineal(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, _, features = cargar_datos(modo="lineal", csv_path=csv)
        for feat in FEATURES_RAW_SOLO_ARBOL:
            assert feat not in features, f"{feat} debería estar excluida (lineal)"

    def test_raw_features_estan_en_arbol(self, df_artist_features_minimal, tmp_path):
        """Las versiones raw SÍ deben estar en modo árbol (solo se eliminan en lineal)."""
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        _, _, features = cargar_datos(modo="arbol", csv_path=csv)
        # lfm_oyentes es raw y debe estar en árbol
        assert "lfm_oyentes" in features

    def test_n_filas_igual_al_csv(self, df_artist_features_minimal, tmp_path):
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        X, y, _ = cargar_datos(modo="arbol", csv_path=csv)
        assert len(X) == len(df_artist_features_minimal)
        assert len(y) == len(df_artist_features_minimal)

    def test_modo_lineal_aplica_escalado(self, df_artist_features_minimal, tmp_path):
        """RobustScaler centra las features; la media en lineal debe diferir de la de arbol."""
        csv = tmp_path / "features.csv"
        df_artist_features_minimal.to_csv(csv, index=False)
        X_arbol,  _, _ = cargar_datos(modo="arbol",  csv_path=csv)
        X_lineal, _, _ = cargar_datos(modo="lineal", csv_path=csv)
        # La escala es diferente: la media global cambia con RobustScaler
        media_arbol  = X_arbol.mean().mean()
        media_lineal = X_lineal.mean().mean()
        assert abs(media_arbol - media_lineal) > 0.01  # no son iguales tras escalar
