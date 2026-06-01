"""
Página 1 — Resultados del proyecto.

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
        'letter-spacing:0.12em;color:#7A7290;margin:8px 0 6px;">Validación del análisis</div>'
        '<div style="font-family:\'JetBrains Mono\';font-size:11.5px;color:#B6ADCB;line-height:1.8;'
        'background:#0F0A1B;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:10px 12px;">'
        'Método: <b style="color:#C6F432;">5 pruebas independientes</b><br>'
        'Artistas: <b style="color:#C6F432;">182 en la base de datos</b><br>'
        'Configuraciones probadas: <b style="color:#C6F432;">100</b><br>'
        'Total de entrenamientos: <b style="color:#C6F432;">500</b>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    page_header(
        crumb="Inicio · <b style='color:#B6ADCB;'>Resultados</b>",
        title_html="Resultados del <em>análisis</em>",
        sub="Comparativa de 8 métodos sobre los 182 artistas de la base de datos.",
        pill_text="Modelo seleccionado",
        pill_color="violet",
    ),
    unsafe_allow_html=True,
)

# KPI strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("Acierto", "68.2 %", "+5 puntos vs versión básica")
m2.metric("Fiabilidad", "66.6 %", "+6 puntos vs versión básica")
m3.metric("Estabilidad", "±6.3 %", "−2.6 pts (más estable)", delta_color="inverse")
m4.metric("Pruebas realizadas", "500", "100 configuraciones × 5 rondas")

divider()

# ---------------------------------------------------------------------------
# 1. Benchmark de modelos
# ---------------------------------------------------------------------------
section_label("01 · COMPARATIVA DE MÉTODOS", icon="leaderboard")
st.markdown("##### Fiabilidad con 5 pruebas independientes · la banda muestra la variación entre pruebas")

_img("comparativa_modelos.png")

st.markdown("""
| Método | Indicadores | Acierto | Fiabilidad | Estabilidad |
|--------|-------------|---------|------------|-------------|
| Referencia: siempre predice "Bajo" | —  | 39.5 % ± 0.9 % | 18.9 % ± 0.3 %  | —  |
| Referencia: predicción aleatoria   | —  | 37.0 % ± 6.9 % | 33.6 % ± 7.2 %  | —  |
| Regresión Logística       | 23 | 55.4 % ± 3.3 % | 55.4 % ± 4.8 %  | ✅ |
| Regresión Ordinal         | 23 | 60.5 % ± 3.9 % | 57.8 % ± 7.5 %  | ✅ |
| SVM                       | 23 | 59.3 % ± 6.2 % | 57.4 % ± 8.1 %  | ⚠️ |
| LightGBM                  | 31 | 60.5 % ± 8.9 % | 58.8 % ± 9.8 %  | ❌ |
| Random Forest             | 31 | 62.4 % ± 2.9 % | 61.0 % ± 3.8 %  | ✅ |
| **XGBoost (versión básica)**   | **31** | **63.0 % ± 8.3 %** | **61.6 % ± 9.4 %** | ❌ |
| **XGBoost (versión optimizada)** 🏆 | **31** | **68.2 % ± 6.7 %** | **66.6 % ± 6.3 %** | ✅ |
""")

divider()

# ---------------------------------------------------------------------------
# 2. Base vs tuned
# ---------------------------------------------------------------------------
section_label("02 · XGBOOST — VERSIÓN BÁSICA vs VERSIÓN OPTIMIZADA", icon="tune")
st.markdown("##### Se probaron 100 configuraciones distintas, cada una validada 5 veces — optimizado para máxima fiabilidad")

_img("xgb_base_vs_tuned.png")

st.info(
    "La versión optimizada aprende de forma más cautelosa y aplica más restricciones internas "
    "para evitar memorizar los datos de entrenamiento. El resultado: más estable y generalizable "
    "con una base de datos de 182 artistas.",
    icon=":material/lightbulb:",
)

divider()

# ---------------------------------------------------------------------------
# 3. Matriz de confusión
# ---------------------------------------------------------------------------
section_label("03 · ACIERTOS Y ERRORES POR NIVEL", icon="grid_on")
st.markdown("##### Cada predicción se hizo sin que el sistema hubiera visto antes ese artista")

col_cm, col_info = st.columns([1, 1])
with col_cm:
    _img("confusion_matrix.png")
with col_info:
    st.markdown(f'#### Análisis por nivel de sala')

    coral = PALETTE["coral"]; amber = PALETTE["amber"]; mint = PALETTE["mint"]
    st.markdown(f"""
<div style="background:rgba(255,92,122,0.08);border:1px solid rgba(255,92,122,0.25);border-radius:14px;padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{coral};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{coral};box-shadow:0 0 8px {coral};"></span>
    BAJO · Fiabilidad: 75 %
  </div>
  <p style="font-size:13.5px;color:#B6ADCB;margin:6px 0 0;">El nivel más fácil de identificar. Presencia digital baja = artista emergente. Perfil muy diferenciado.</p>
</div>

<div style="background:rgba(255,181,71,0.08);border:1px solid rgba(255,181,71,0.25);border-radius:14px;padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{amber};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{amber};box-shadow:0 0 8px {amber};"></span>
    MEDIO · Fiabilidad: 55 %
  </div>
  <p style="font-size:13.5px;color:#B6ADCB;margin:6px 0 0;">El nivel más difícil — zona de incertidumbre: 13 se confunden con Bajo, 13 con Alto.</p>
</div>

<div style="background:rgba(0,224,164,0.08);border:1px solid rgba(0,224,164,0.25);border-radius:14px;padding:14px 16px;">
  <div style="display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono';font-size:11px;color:{mint};text-transform:uppercase;letter-spacing:0.1em;">
    <span style="width:8px;height:8px;border-radius:50%;background:{mint};box-shadow:0 0 8px {mint};"></span>
    ALTO · Fiabilidad: 55 %
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

Los errores extremos (Alto ↔ Bajo) suman solo **7 casos** — el sistema rara vez confunde los extremos.
""")

divider()

# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------
section_label("04 · ¿QUÉ DATOS IMPORTAN MÁS?", icon="bar_chart")

tab_top10, tab_rf, tab_xgb, tab_shap = st.tabs([
    "Top 10 · Resumen", "Random Forest", "XGBoost", "Desglose detallado",
])

with tab_top10:
    _img("feature_importance_top10.png")
    st.markdown("""
**Los dos análisis coinciden en estas prioridades:**

1. Nº de conciertos registrados — el dato más importante ⚠️ *(41 % de artistas sin datos, se asume 0)*
2. Oyentes en Last.fm — el dato más fiable (relación directa con el nivel real)
3. Vistas en YouTube / Suscriptores en YouTube — volumen digital acumulado
4. Países donde ha actuado / Escuchas en Last.fm — alcance geográfico y fidelidad histórica

Los datos de tendencias recientes tienen un peso menor.
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
            _img("shap_summary_bar.png", caption="Peso medio de cada dato sobre el resultado final")
        with col_bee:
            _img("shap_beeswarm_alto.png", caption="Gráfico de puntos — Nivel ALTO")

        st.markdown("""
**Cómo leer estos gráficos**

- **Gráfico de barras:** peso global de cada dato — cuanto más larga la barra, más influye ese dato en el resultado final.

- **Gráfico de puntos:** cada punto es un artista. Hacia la derecha = empuja hacia nivel ALTO. Hacia la izquierda = empuja hacia niveles menores. Color rojo = valor alto del dato, azul = valor bajo.

> Para ver el desglose de un artista concreto, ve a **Predice un artista** — se genera automáticamente.
""")
    else:
        st.info(
            "Los gráficos de impacto aún no están listos. "
            "Contacta con el administrador para regenerarlos.",
            icon=":material/info:",
        )
