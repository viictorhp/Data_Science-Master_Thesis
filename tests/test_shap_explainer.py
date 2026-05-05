"""
test_shap_explainer.py
======================
Tests de src/models/shap_explainer.py

Se mockean el modelo y el explainer de SHAP para no depender de
xgb_tuned.joblib (gitignoreado) ni de datos de entrenamiento.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import matplotlib
matplotlib.use("Agg")  # backend sin ventana gráfica para tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.models.shap_explainer import LABEL_MAP


# ===========================================================================
# Tests de constantes
# ===========================================================================

class TestConstantes:

    def test_label_map_tiene_tres_entradas(self):
        assert len(LABEL_MAP) == 3

    def test_label_map_indices_correctos(self):
        assert LABEL_MAP[0] == "bajo"
        assert LABEL_MAP[1] == "medio"
        assert LABEL_MAP[2] == "alto"

    def test_label_map_valores_validos(self):
        assert set(LABEL_MAP.values()) == {"bajo", "medio", "alto"}


# ===========================================================================
# Tests de shap_waterfall_fig()
# ===========================================================================

class TestSHAPWaterfallFig:
    """
    shap_waterfall_fig() se testa mockeando _cargar_explainer y
    shap.plots.waterfall para evitar dependencias externas.
    """

    def _build_mock_explainer(self, mock_model):
        """Construye un explainer mockeado que simula shap.TreeExplainer."""
        model, meta = mock_model
        n_features = len(meta["features"])

        # sv[0, :, clase_idx] → Explanation para una muestra y una clase
        sv_sample = MagicMock()
        sv_sample.values      = np.zeros(n_features)
        sv_sample.base_values = np.float64(0.5)
        sv_sample.data        = np.zeros(n_features)
        sv_sample.feature_names = meta["features"]

        sv = MagicMock()
        sv.__getitem__ = MagicMock(return_value=sv_sample)

        explainer = MagicMock()
        explainer.return_value = sv
        explainer.model        = model   # model.predict_proba → [0.2, 0.6, 0.2]

        return explainer, meta

    def test_retorna_figura_matplotlib(self, mock_model):
        explainer, meta = self._build_mock_explainer(mock_model)
        features_dict = {feat: 0.0 for feat in meta["features"]}

        fig_fake = plt.figure()
        with patch("src.models.shap_explainer._cargar_explainer", return_value=(explainer, meta)), \
             patch("shap.plots.waterfall"), \
             patch("matplotlib.pyplot.gcf", return_value=fig_fake), \
             patch("matplotlib.pyplot.title"), \
             patch("matplotlib.pyplot.tight_layout"):

            from src.models.shap_explainer import shap_waterfall_fig
            fig, clase = shap_waterfall_fig(features_dict)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_retorna_clase_valida(self, mock_model):
        explainer, meta = self._build_mock_explainer(mock_model)
        features_dict = {feat: 0.0 for feat in meta["features"]}

        fig_fake = plt.figure()
        with patch("src.models.shap_explainer._cargar_explainer", return_value=(explainer, meta)), \
             patch("shap.plots.waterfall"), \
             patch("matplotlib.pyplot.gcf", return_value=fig_fake), \
             patch("matplotlib.pyplot.title"), \
             patch("matplotlib.pyplot.tight_layout"):

            from src.models.shap_explainer import shap_waterfall_fig
            _, clase = shap_waterfall_fig(features_dict)

        assert clase in {"bajo", "medio", "alto"}
        plt.close("all")

    def test_clase_corresponde_al_argmax_de_proba(self, mock_model):
        """predict_proba devuelve [0.2, 0.6, 0.2] → debe predecir 'medio' (índice 1)."""
        explainer, meta = self._build_mock_explainer(mock_model)
        features_dict = {feat: 0.0 for feat in meta["features"]}

        fig_fake = plt.figure()
        with patch("src.models.shap_explainer._cargar_explainer", return_value=(explainer, meta)), \
             patch("shap.plots.waterfall"), \
             patch("matplotlib.pyplot.gcf", return_value=fig_fake), \
             patch("matplotlib.pyplot.title"), \
             patch("matplotlib.pyplot.tight_layout"):

            from src.models.shap_explainer import shap_waterfall_fig
            _, clase = shap_waterfall_fig(features_dict)

        assert clase == "medio"
        plt.close("all")

    def test_se_llama_a_shap_plots_waterfall(self, mock_model):
        """Verificar que se invoca shap.plots.waterfall (integración con librería SHAP)."""
        explainer, meta = self._build_mock_explainer(mock_model)
        features_dict = {feat: 0.0 for feat in meta["features"]}

        fig_fake = plt.figure()
        with patch("src.models.shap_explainer._cargar_explainer", return_value=(explainer, meta)), \
             patch("shap.plots.waterfall") as mock_waterfall, \
             patch("matplotlib.pyplot.gcf", return_value=fig_fake), \
             patch("matplotlib.pyplot.title"), \
             patch("matplotlib.pyplot.tight_layout"):

            from src.models.shap_explainer import shap_waterfall_fig
            shap_waterfall_fig(features_dict)

        mock_waterfall.assert_called_once()
        plt.close("all")
