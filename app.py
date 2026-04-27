import streamlit as st

st.set_page_config(
    page_title="Predicción Tier de Sala · TFM",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎵 Predictor de Tier de Sala")
st.caption("Rap y música urbana española · TFM 2026")

st.markdown("""
Herramienta de Machine Learning para estimar el **tamaño de sala** que un artista
de rap/urbano español puede llenar, a partir de sus métricas de presencia digital
y actividad en directo.
""")

st.divider()

# ---------------------------------------------------------------------------
# Métricas de resumen
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Artistas en el dataset", "157")
col2.metric("Accuracy del modelo", "67.5%")
col3.metric("F1 macro", "66.9%")
col4.metric("Features utilizadas", "31")

st.divider()

# ---------------------------------------------------------------------------
# Tarjetas de navegación
# ---------------------------------------------------------------------------
st.subheader("¿Qué quieres hacer?")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
**📊 Resultados del proyecto**

Benchmark de modelos, comparativa base vs tuned,
feature importance y matriz de confusión.
""")

with c2:
    st.success("""
**🎤 Predice un artista**

Introduce las métricas básicas de un artista
y obtén su tier de sala predicho con probabilidades.
""")

with c3:
    st.warning("""
**🤖 Análisis IA**

El agente LangChain explica la predicción
en lenguaje natural y responde tus preguntas.
""")

st.divider()

# ---------------------------------------------------------------------------
# Contexto del problema
# ---------------------------------------------------------------------------
st.subheader("El problema")

st.markdown("""
| Tier | Nivel | Aforo equivalente | Ejemplos |
|------|-------|-------------------|---------|
| 1 | **Bajo** | < 200 personas | Sala Víbora, artistas sin sala propia |
| 2 | **Medio** | 200 – 2.000 personas | Planta Baja, La Riviera, Razzmatazz |
| 3 | **Alto** | > 2.000 personas | WiZink, Movistar Arena, festivales |

**Distribución del dataset**: bajo = 62 artistas · medio = 57 · alto = 38 · **total = 157**
""")
