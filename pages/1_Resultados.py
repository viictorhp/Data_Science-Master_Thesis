from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Resultados · TFM",
    page_icon="📊",
    layout="wide",
)

FIGURES = Path("reports/figures")

st.title("📊 Resultados del proyecto")
st.caption("Benchmark de modelos · StratifiedKFold k=5 · 157 artistas")

# ---------------------------------------------------------------------------
# 1. Benchmark de modelos
# ---------------------------------------------------------------------------
st.subheader("1. Comparativa de modelos")
st.caption("Todos los modelos evaluados con CV5 estratificado sobre los 157 artistas.")

st.image(str(FIGURES / "comparativa_modelos.png"), width='stretch')

st.markdown("""
| Modelo | Features | Accuracy (CV5) | F1 macro (CV5) | Estabilidad |
|--------|----------|---------------|----------------|-------------|
| Dummy most_frequent | — | 39.5% ± 0.9% | 18.9% ± 0.3% | — |
| Dummy stratified | — | 37.0% ± 6.9% | 33.6% ± 7.2% | — |
| Regresión Logística | 23 lineal | 55.4% ± 3.3% | 55.4% ± 4.8% | ✅ |
| Regresión Ordinal (mord) | 23 lineal | 60.5% ± 3.9% | 57.8% ± 7.5% | ✅ |
| SVM (RBF) | 23 lineal | 59.3% ± 6.2% | 57.4% ± 8.1% | ⚠️ |
| LightGBM | 31 árbol | 60.5% ± 8.9% | 58.8% ± 9.8% | ❌ |
| Random Forest | 31 árbol | 62.4% ± 2.9% | 61.0% ± 3.8% | ✅ |
| **XGBoost (base)** | **31 árbol** | **63.0% ± 8.3%** | **61.6% ± 9.4%** | ❌ |
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Ajuste de hiperparámetros
# ---------------------------------------------------------------------------
st.subheader("2. XGBoost — Base vs Tuned")
st.caption("RandomizedSearchCV · 100 iteraciones × CV5 = 500 fits · optimizado por F1 macro")

st.image(str(FIGURES / "xgb_base_vs_tuned.png"), width='stretch')

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy",  "67.5%", "+4.5 pts vs base")
col2.metric("F1 macro",  "66.9%", "+5.3 pts vs base")
col3.metric("Std F1",    "±3.7%", "-5.7 pts vs base", delta_color="inverse")

st.markdown("""
**Parámetros clave del modelo tuned**: `learning_rate=0.01` · `min_child_weight=5` ·
`reg_alpha=1.0` · `reg_lambda=2.0` · `n_estimators=400` · `max_depth=4`

> Todos los cambios apuntan a **más regularización** — el modelo tuned es más conservador
> y estable, ideal para un dataset de 157 artistas.
""")

st.divider()

# ---------------------------------------------------------------------------
# 3. Matriz de confusión
# ---------------------------------------------------------------------------
st.subheader("3. Matriz de confusión — XGBoost (predicciones out-of-fold)")
st.caption("Cada artista predicho por un modelo que no lo vio en entrenamiento (OOF).")

col_cm, col_info = st.columns([1, 1])
with col_cm:
    st.image(str(FIGURES / "confusion_matrix.png"), width='stretch')
with col_info:
    st.markdown("""
**Análisis por clase:**

- 🔴 **Bajo** (F1 = 0.75) — La clase mejor clasificada.
  Señal digital baja = artista underground. Relativamente fácil.

- 🟡 **Medio** (F1 = 0.55) — La más difícil.
  Zona fronteriza: 13 clasificados como bajo, 13 como alto.

- 🟢 **Alto** (F1 = 0.55) — 14 artistas mal clasificados como medio.
  El modelo infravalora artistas con presencia digital moderada
  pero con gran convocatoria (Yung Beef, Rels B, SAIKO, Maka...).

**Pares de error más frecuentes:**

| Error | Casos |
|-------|-------|
| alto → medio | 14 |
| medio → bajo | 13 |
| medio → alto | 13 |
| bajo → medio | 11 |

Los errores de 2 saltos (alto→bajo, bajo→alto) son solo 7 casos —
el modelo respeta la ordinalidad implícitamente.
""")

st.divider()

# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------
st.subheader("4. Feature importance")

tab_top10, tab_rf, tab_xgb = st.tabs(["Top 10 comparativa", "Random Forest (MDI)", "XGBoost (Gain)"])

with tab_top10:
    st.image(str(FIGURES / "feature_importance_top10.png"), width='stretch')
    st.markdown("""
**Consenso RF + XGBoost:**

1. `sl_num_conciertos` — #1 en ambos modelos ⚠️ *(41% NaN imputados a 0)*
2. `lfm_oyentes` — señal más limpia del dataset (Spearman r = +0.63)
3. `yt_vistas_totales` / `yt_suscriptores` — volumen digital acumulado
4. `sl_num_paises` / `lfm_scrobbles` — alcance geográfico y engagement histórico

Las features de tendencias recientes tienen importancia secundaria.
""")

with tab_rf:
    st.image(str(FIGURES / "feature_importance_rf.png"), width='stretch')

with tab_xgb:
    st.image(str(FIGURES / "feature_importance_xgb.png"), width='stretch')
