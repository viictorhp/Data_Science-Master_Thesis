"""
Home — Predictor de Tier de Sala (TFM 2026).

Rediseño visual con identidad propia ("Studio"):
  · Fondo nocturno violeta + acentos magenta + lima eléctrica
  · Fuentes: Space Grotesk (titulares) + Inter + JetBrains Mono
  · Iconos modernos (Material Symbols) en lugar de emojis genéricos

La lógica de modelo es la misma del proyecto original — sólo cambia la UI.
"""

import json
from pathlib import Path

import streamlit as st
from styles import (
    inject_styles, page_header, hero, nav_card,
    brand_block, section_label, divider, tier_chip,
)

# ---------------------------------------------------------------------------
# Carga dinámica de metadatos del modelo
# ---------------------------------------------------------------------------
def _load_meta() -> dict:
    meta_path = Path(__file__).parent / "models" / "metadata.json"
    try:
        with open(meta_path, encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return {}

_META = _load_meta()
_ACC   = f"{_META.get('cv5_metrics', {}).get('acc_mean', 0.682) * 100:.1f} %"
_F1    = f"{_META.get('cv5_metrics', {}).get('f1_mean',  0.666) * 100:.1f} %"
_ACC_D = f"+{(_META.get('cv5_metrics', {}).get('acc_mean', 0.682) - 0.395) * 100:.1f} pts vs base"
_F1_D  = f"+{(_META.get('cv5_metrics', {}).get('f1_mean',  0.666) - 0.189) * 100:.1f} pts vs base"
_N     = str(_META.get("n_samples", 182))
_NFEAT = str(_META.get("n_features", 32))
_DIST  = _META.get("class_distribution", {"bajo": 81, "medio": 62, "alto": 39})

st.set_page_config(
    page_title="Tier Predictor · TFM",
    page_icon="🎚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ---------------------------------------------------------------------------
# Sidebar — branding + estado del modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(brand_block(), unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\';font-size:10px;'
        'text-transform:uppercase;letter-spacing:0.12em;color:#7A7290;margin:8px 0 6px;">Estado</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:#B6ADCB;'
        'background:rgba(0,224,164,0.08);border:1px solid rgba(0,224,164,0.25);'
        'border-radius:10px;padding:8px 12px;margin-bottom:8px;font-family:\'JetBrains Mono\';">'
        '<span style="width:8px;height:8px;border-radius:50%;background:#00E0A4;box-shadow:0 0 8px #00E0A4;"></span>'
        'modelo online · XGBoost tuned</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\';font-size:11px;color:#7A7290;line-height:1.7;'
        f'padding:8px 4px;">'
        f'<b style="color:#B6ADCB;">{_N}</b> artistas · <b style="color:#B6ADCB;">{_NFEAT}</b> features<br>'
        f'<b style="color:#B6ADCB;">Acc {_ACC}</b> · <b style="color:#B6ADCB;">F1 {_F1}</b><br>'
        f'CV5 estratificado · k=5'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    page_header(
        crumb="Inicio",
        title_html="Predice qué <em>sala llena</em><br>un artista de rap español",
        sub=("Modelo de Machine Learning entrenado sobre 182 artistas del panorama urbano nacional. "
             "Estima el tier de sala a partir de su huella digital y actividad en directo."),
        pill_text="Modelo online",
        pill_color="mint",
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    hero(
        eyebrow="TFM · Data Science · 2026",
        title_html="De métricas a <em style=\"color:#C6F432;font-style:normal;\">aforos</em>.<br>En un solo click.",
        lead=("Introduce Spotify, Last.fm, YouTube y setlist.fm. Recibe un tier "
              "(bajo · medio · alto) con probabilidades, alertas e interpretación SHAP. "
              "Después, pregunta al agente LangChain."),
    ),
    unsafe_allow_html=True,
)

c_a, c_b, _ = st.columns([1, 1, 2])
with c_a:
    if st.button(":material/play_arrow: Probar con un artista", use_container_width=True):
        st.switch_page("pages/2_Prediccion.py")
with c_b:
    if st.button(":material/analytics: Ver resultados del modelo", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_Resultados.py")

# ---------------------------------------------------------------------------
# Métricas resumen
# ---------------------------------------------------------------------------
section_label("Resumen del modelo", icon="dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Artistas dataset", _N, f"Bajo {_DIST['bajo']} · Med {_DIST['medio']} · Alto {_DIST['alto']}")
m2.metric("Accuracy CV5", _ACC, _ACC_D)
m3.metric("F1 macro", _F1, _F1_D)
m4.metric("Features", _NFEAT, "4 fuentes · 8 ratios")

# ---------------------------------------------------------------------------
# Tarjetas de navegación
# ---------------------------------------------------------------------------
section_label("¿Qué quieres hacer?", icon="arrow_forward")
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown(nav_card(
        "01", "insights", "Resultados del proyecto",
        "Benchmark de 8 modelos, comparativa base vs tuned, feature importance + SHAP global y matriz de confusión OOF.",
        accent="violet",
    ), unsafe_allow_html=True)
    if st.button("Explorar resultados", key="nav_res", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_Resultados.py")

with n2:
    st.markdown(nav_card(
        "02", "mic_external_on", "Predice un artista",
        "Introduce métricas básicas de Spotify, Last.fm, YouTube y setlist.fm. "
        "Devuelve tier, probabilidades, alertas e interpretación.",
        accent="lime",
    ), unsafe_allow_html=True)
    if st.button("Lanzar predicción", key="nav_pred", use_container_width=True):
        st.switch_page("pages/2_Prediccion.py")

with n3:
    st.markdown(nav_card(
        "03", "auto_awesome", "Análisis con IA",
        "El agente LangChain + Groq (Llama 3.3 70B) explica la predicción en lenguaje natural "
        "y responde a tus preguntas.",
        accent="pink",
    ), unsafe_allow_html=True)
    if st.button("Conversar con el agente", key="nav_ai", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_Analisis_IA.py")

divider()

# ---------------------------------------------------------------------------
# El problema — tabla de tiers
# ---------------------------------------------------------------------------
section_label("El problema", icon="diamond")
st.markdown("Tres tiers de sala definidos por aforo y contexto del circuito urbano español.")

TIERS_INFO = [
    ("bajo",  "01", "< 200 personas",  "Sala Víbora, locales pequeños, artistas sin sala propia", str(_DIST["bajo"])),
    ("medio", "02", "200 – 2 000",     "Planta Baja, La Riviera, Razzmatazz",                     str(_DIST["medio"])),
    ("alto",  "03", "> 2 000",         "WiZink, Movistar Arena, festivales (Sonorama, RBF…)",     str(_DIST["alto"])),
]
for nivel, num, aforo, ejemplos, n in TIERS_INFO:
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:120px 110px 1fr auto;
                    align-items:center;gap:16px;
                    background:var(--surface);border:1px solid var(--line);
                    border-radius:14px;padding:14px 18px;margin-bottom:8px;">
          {tier_chip(nivel)}
          <span style="font-family:'JetBrains Mono';font-size:11px;color:#7A7290;">
            {num} · {aforo}
          </span>
          <span style="font-size:13px;color:#B6ADCB;">{ejemplos}</span>
          <span style="font-family:'JetBrains Mono';font-size:13px;font-weight:600;
                       color:#F2EEFA;text-align:right;">n={n}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f'<p style="font-family:\'JetBrains Mono\';font-size:11px;color:#7A7290;margin-top:6px;">'
    f'{_N} artistas · clase mayoritaria {int(_DIST["bajo"]) / int(_N) * 100:.1f} % (referencia para baseline).</p>',
    unsafe_allow_html=True,
)
