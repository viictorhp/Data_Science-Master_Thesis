"""
Página 1 — Resultados del proyecto.

Mismo contenido que la versión original (benchmark, base vs tuned, matriz de confusión,
feature importance, SHAP) — con el sistema de diseño Studio aplicado.
"""

from pathlib import Path
import streamlit as st

from styles import (
    inject_styles, page_header, brand_block,
    section_label, divider, PALETTE,
)

st.set_page_config(page_title="Resultados · TFM", page_icon="📊", layout="wide")
inject_styles()

FIGURES = Path(__file__).parent.parent / "reports" / "figures"


def _img(filename: str, caption: str = "", **kwargs) -> None:
    """Muestra una imagen con manejo de error si el archivo no existe."""
    p = FIGURES / filename
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True, **kwargs)
    else:
        st.info(
            f"Imagen `{filename}` no encontrada — ejecuta el pipeline para regenerarla.",
            icon=":material/image_not_supported:",
        )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(brand_block(), unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\';font-size:10px;text-transform:uppercase;'
        'letter-spacing:0.12em;color:#7A7290;margin:8px 0 6px;">Configuración eval</div>'
        '<div style="font-family:\'JetBrains Mono\';font-size:11.5px;color:#B6ADCB;line-height:1.8;'
        'background:#0F0A1B;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:10px 12px;">'
        'cv: <b style="color:#C6F432;">StratifiedKFold(5)</b><br>'
        'metric: <b style="color:#C6F432;">f1_macro</b><br>'
        'search: <b style="color:#C6F432;">100 iters</b><br>'
        'fits: <b style="color:#C6F432;">500</b>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    page_header(
        crumb="Inicio · <b style='color:#B6ADCB;'>Resultados</b>",
        title_html="Resultados del <em>modelo</em>",
        sub="Benchmark de 8 algoritmos con StratifiedKFold k=5 sobre los 182 artistas del dataset.",
        pill_text="XGBoost · tuned",
        pill_color="violet",
    ),
    unsafe_allow_html=True,
)

# KPI strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy CV5", "68.2 %", "+5.0 pts vs base")
m2.metric("F1 macro", "66.6 %", "+6.3 pts vs base")
m3.metric("Std F1", "±6.3 %", "−2.6 pts (más estable)", delta_color="inverse")
m4.metric("Fits ejecutados", "500", "100 iters · CV5")

divider()

# ---------------------------------------------------------------------------
# 1. Benchmark de modelos
# ---------------------------------------------------------------------------
section_label("01 · COMPARATIVA DE MODELOS", icon="leaderboard")
st.markdown("##### F1 macro con StratifiedKFold k=5 · banda = desviación estándar")

_img("comparativa_modelos.png")

st.markdown("""
| Modelo | Features | Accuracy CV5 | F1 macro CV5 | Estabilidad |
|--------|----------|--------------|--------------|-------------|
| Dummy `most_frequent`     | —            | 39.5 % ± 0.9 % | 18.9 % ± 0.3 %  | —  |
| Dummy `stratified`        | —            | 37.0 % ± 6.9 % | 33.6 % ± 7.2 %  | —  |
| Regresión Logística       | 23 lineal    | 55.4 % ± 3.3 % | 55.4 % ± 4.8 %  | ✅ |
| Regresión Ordinal (mord)  | 23 lineal    | 60.5 % ± 3.9 % | 57.8 % ± 7.5 %  | ✅ |
| SVM (RBF)                 | 23 lineal    | 59.3 % ± 6.2 % | 57.4 % ± 8.1 %  | ⚠️ |
| LightGBM                  | 31 árbol     | 60.5 % ± 8.9 % | 58.8 % ± 9.8 %  | ❌ |
| Random Forest             | 31 árbol     | 62.4 % ± 2.9 % | 61.0 % ± 3.8 %  | ✅ |
| **XGBoost (base)**        | **31 árbol** | **63.0 % ± 8.3 %** | **61.6 % ± 9.4 %** | ❌ |
| **XGBoost (tuned)** 🏆    | **31 árbol** | **68.2 % ± 6.7 %** | **66.6 % ± 6.3 %** | ✅ |
""")

divider()

# ---------------------------------------------------------------------------
# 2. Base vs tuned
# ---------------------------------------------------------------------------
section_label("02 · XGBOOST — BASE vs TUNED", icon="tune")
st.markdown("##### RandomizedSearchCV · 100 iteraciones × CV5 = 500 fits · optimizado por F1 macro")

_img("xgb_base_vs_tuned.png")

st.info(
    "**Parámetros clave del modelo tuned:** `learning_rate=0.01` · `min_child_weight=10` · "
    "`reg_alpha=1.0` · `reg_lambda=2.0` · `n_estimators=400` · `max_depth=2`\n\n"
    "Todos los cambios apuntan a **más regularización** — el modelo tuned es más conservador "
    "y estable, ideal para un dataset de 182 artistas.",
    icon=":material/lightbulb:",
)

divider()

# ---------------------------------------------------------------------------
# 3. Matriz de confusión
# ---------------------------------------------------------------------------
section_label("03 · MATRIZ DE CONFUSIÓN (OOF)", icon="grid_on")
st.markdown("##### Cada artista predicho por un modelo que no lo vio en entrenamiento")

col_cm, col_info = st.columns([1, 1])
with col_cm:
    _img("confusion_matrix.png")
with col_info:
    st.markdown(f'#### Análisis por clase')

    coral = PALETTE["coral"]; amber = PALETTE["amber"]; mint = PALETTE["mint"]
    st.markdown(f"""
<div style="background:rgba(255,92,122,0.08);border:1px solid rgba(255,92,122,0.25);border-radius:14px;padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{coral};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{coral};box-shadow:0 0 8px {coral};"></span>
    BAJO · F1 = 0.75
  </div>
  <p style="font-size:13.5px;color:#B6ADCB;margin:6px 0 0;">La clase mejor clasificada. Señal digital baja = artista underground. Patrón claro.</p>
</div>

<div style="background:rgba(255,181,71,0.08);border:1px solid rgba(255,181,71,0.25);border-radius:14px;padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{amber};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{amber};box-shadow:0 0 8px {amber};"></span>
    MEDIO · F1 = 0.55
  </div>
  <p style="font-size:13.5px;color:#B6ADCB;margin:6px 0 0;">La más difícil — zona fronteriza: 13 → bajo, 13 → alto.</p>
</div>

<div style="background:rgba(0,224,164,0.08);border:1px solid rgba(0,224,164,0.25);border-radius:14px;padding:14px 16px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{mint};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{mint};box-shadow:0 0 8px {mint};"></span>
    ALTO · F1 = 0.55
  </div>
  <p style="font-size:13.5px;color:#B6ADCB;margin:6px 0 0;">14 mal clasificados como medio. Infravalora artistas con presencia digital moderada pero gran convocatoria (Yung Beef, Rels B, SAIKO, Maka).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("##### Pares de error más frecuentes")
st.markdown("""
| Error            | Casos |
|------------------|-------|
| alto → medio     | 14    |
| medio → bajo     | 13    |
| medio → alto     | 13    |
| bajo → medio     | 11    |

Los errores de 2 saltos (alto → bajo, bajo → alto) suman solo **7 casos** — el modelo respeta la ordinalidad implícitamente.
""")

divider()

# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------
section_label("04 · IMPORTANCIA DE VARIABLES", icon="bar_chart")

tab_top10, tab_rf, tab_xgb, tab_shap = st.tabs([
    "Top 10 consenso", "Random Forest", "XGBoost", "SHAP global",
])

with tab_top10:
    _img("feature_importance_top10.png")
    st.markdown("""
**Consenso RF + XGBoost:**

1. `sl_num_conciertos` — #1 en ambos modelos ⚠️ *(41 % NaN imputados a 0)*
2. `lfm_oyentes` — la señal más limpia del dataset (Spearman r = +0.63)
3. `yt_vistas_totales` / `yt_suscriptores` — volumen digital acumulado
4. `sl_num_paises` / `lfm_scrobbles` — alcance geográfico y engagement histórico

Las features de tendencias recientes tienen importancia secundaria.
""")

with tab_rf:
    _img("feature_importance_rf.png")

with tab_xgb:
    st.image(str(FIGURES / "feature_importance_xgb.png"), use_container_width=True)

with tab_shap:
    SHAP_BAR = FIGURES / "shap_summary_bar.png"
    SHAP_BEE = FIGURES / "shap_beeswarm_alto.png"

    if SHAP_BAR.exists() and SHAP_BEE.exists():
        col_bar, col_bee = st.columns(2)
        with col_bar:
            _img("shap_summary_bar.png", caption="Media |SHAP| global (todas las clases)")
        with col_bee:
            _img("shap_beeswarm_alto.png", caption="Beeswarm — Clase ALTO")

        st.markdown("""
**Cómo leer estos plots**

- **Bar plot:** importancia global de cada feature, medida como la media del
  valor absoluto de sus SHAP values. A diferencia del MDI de Random Forest, SHAP es
  independiente de la estructura del árbol y más consistente ante features correlacionadas.

- **Beeswarm:** cada punto es un artista. El eje X muestra cuánto empuja la feature
  hacia ALTO (positivos) o hacia clases inferiores (negativos). El color indica el valor
  real (rojo = alto, azul = bajo).

> Para la explicación de una predicción individual, ve a **Predice un artista** — el waterfall
> SHAP se genera automáticamente para cada artista analizado.
""")
    else:
        st.info(
            "Los plots SHAP globales aún no se han generado. "
            "Ejecuta `python scripts/generar_shap.py` desde la raíz del proyecto y recarga la página.",
            icon=":material/info:",
        )
