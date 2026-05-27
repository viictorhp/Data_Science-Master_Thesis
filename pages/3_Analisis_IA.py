"""
Página 3 — Análisis con IA.

Misma lógica original (agentes LangChain + Groq) con el sistema de diseño Studio.
"""

import sys
sys.path.insert(0, '.')

import streamlit as st

from src.agents.explicador import chat, generar_explicacion, _system_prompt
from src.utils.log_streamlit import log, render_sidebar_log, reset_prediccion

from styles import (
    inject_styles, page_header, brand_block, section_label, divider,
    TIER_COLOR, PALETTE,
)

st.set_page_config(page_title="Análisis IA · TFM", page_icon="🤖", layout="wide")
inject_styles()

with st.sidebar:
    st.markdown(brand_block(), unsafe_allow_html=True)
    render_sidebar_log()

# ---------------------------------------------------------------------------
# Guardia: necesita predicción previa
# ---------------------------------------------------------------------------
if "prediccion" not in st.session_state:
    st.markdown(
        page_header(
            crumb="Inicio · <b style='color:#B6ADCB;'>Análisis IA</b>",
            title_html="Análisis con <em>IA</em>",
            sub="El agente LangChain + Groq explica la predicción en lenguaje natural.",
            pill_text="Esperando predicción",
            pill_color="amber",
        ),
        unsafe_allow_html=True,
    )
    st.warning(
        "Aún no hay ninguna predicción. Ve primero a **Predicción**, rellena el formulario y pulsa *Predecir*.",
        icon=":material/info:",
    )
    if st.button(":material/mic_external_on: Ir a Predicción", type="primary"):
        st.switch_page("pages/2_Prediccion.py")
    st.stop()

pred      = st.session_state["prediccion"]
resultado = pred["resultado"]
nombre    = pred["nombre"]
info      = pred["info_conciertos"]
nivel     = resultado["nivel"]
proba     = resultado["probabilidades"]
features  = resultado["features"]

color = TIER_COLOR[nivel]
conf_pct = int(round(proba[nivel] * 100))
segundo  = sorted(proba.items(), key=lambda x: x[1], reverse=True)[1]
seg_color = TIER_COLOR[segundo[0]]

# ---------------------------------------------------------------------------
# Cabecera con callout grande
# ---------------------------------------------------------------------------
st.markdown(
    page_header(
        crumb="Inicio · <b style='color:#B6ADCB;'>Análisis IA</b>",
        title_html="Conversa con el <em>agente</em>",
        sub=("LangChain + Groq · Llama 3.3 70B · contexto del modelo y la predicción "
             "inyectado en el system prompt."),
        pill_text="Groq online",
        pill_color="mint",
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tarjeta de resumen de la predicción activa
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background:var(--surface);border:1px solid var(--line);
                border-radius:18px;padding:20px 26px;
                display:grid;grid-template-columns:auto 1fr auto auto;gap:28px;align-items:center;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:48px;height:48px;border-radius:14px;
                    background:conic-gradient({color} 0% {conf_pct}%, rgba(255,255,255,0.08) {conf_pct}% 100%);
                    display:flex;align-items:center;justify-content:center;">
          <div style="width:34px;height:34px;border-radius:50%;background:var(--surface);
                      display:flex;align-items:center;justify-content:center;
                      font-family:'JetBrains Mono';font-size:11px;font-weight:600;color:{color};">{conf_pct}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono';font-size:10.5px;color:#7A7290;
                      text-transform:uppercase;letter-spacing:0.1em;">Artista</div>
          <div style="font-family:'Space Grotesk';font-size:18px;font-weight:600;">{nombre}</div>
        </div>
      </div>
      <div>
        <div style="font-family:'JetBrains Mono';font-size:10.5px;color:#7A7290;
                    text-transform:uppercase;letter-spacing:0.1em;">Tier predicho</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:2px;">
          <span style="width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 8px {color};"></span>
          <b style="font-family:'Space Grotesk';font-size:18px;color:{color};">{nivel.upper()}</b>
        </div>
      </div>
      <div>
        <div style="font-family:'JetBrains Mono';font-size:10.5px;color:#7A7290;
                    text-transform:uppercase;letter-spacing:0.1em;">Confianza</div>
        <div style="font-family:'Space Grotesk';font-size:22px;font-weight:600;color:{color};">{proba[nivel]:.1%}</div>
      </div>
      <div>
        <div style="font-family:'JetBrains Mono';font-size:10.5px;color:#7A7290;
                    text-transform:uppercase;letter-spacing:0.1em;">2º más probable</div>
        <div style="font-family:'Space Grotesk';font-size:14px;">
          <span style="color:{seg_color};">{segundo[0]}</span>
          <span style="color:#7A7290;font-family:'JetBrains Mono';font-size:13px;">· {segundo[1]:.1%}</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

divider()

# ---------------------------------------------------------------------------
# Explicación inicial (se genera una vez por predicción)
# ---------------------------------------------------------------------------
pred_id = id(resultado)
if "explicacion_inicial" not in st.session_state or \
   st.session_state.get("prediccion_id") != pred_id:

    st.session_state["prediccion_id"] = pred_id

    with st.status(":material/bolt: Consultando agente IA…", expanded=True) as status:
        st.write("**1 · Construcción del contexto del sistema**")
        system_prompt = _system_prompt(resultado, nombre, info)
        st.write(f"  Modelo LLM   : `llama-3.3-70b-versatile` (Groq)")
        st.write(f"  Temperatura  : `0.3`")
        st.write(f"  Artista      : {nombre}")
        st.write(f"  Tier predicho: {nivel.upper()} ({proba[nivel]:.1%})")
        st.write(f"  Métricas incluidas en el prompt:")
        tiene_sl = int(features.get("sl_tiene_datos", 0))
        st.write(f"    • Spotify  : {features['sp_num_albums']:.0f} álbumes · {features['sp_num_singles']:.0f} singles · {features['sp_anos_activo']:.1f} años")
        st.write(f"    • Last.fm  : {features['lfm_oyentes']:,.0f} oyentes · {features['lfm_scrobbles']:,.0f} scrobbles")
        st.write(f"    • YouTube  : {features['yt_suscriptores']:,.0f} subs · {features['yt_vistas_totales']:,.0f} vistas")
        st.write(f"    • Directo  : {features['sl_num_conciertos']:.0f} conciertos · setlist.fm: {'sí' if tiene_sl else 'no aparece'}")
        st.write(f"    • Tendencias: Google {features.get('trend_gtrends_interes_medio', 0):.1f}/100 · YT rec. {features.get('trend_yt_vistas_recientes', 0):,.0f} vistas")
        st.write(f"  Info adicional: {'Sí' if info else 'No'}")
        st.write(f"  Longitud del system prompt: {len(system_prompt)} caracteres")
        log(f"System prompt construido para {nombre}: {len(system_prompt)} chars", "API")

        with st.expander(":material/terminal:  Ver system prompt completo enviado a la IA"):
            st.code(system_prompt, language=None)

        st.write("**2 · Enviando petición a Groq**")
        log("Llamando a Groq API — generar_explicacion()", "API")
        try:
            explicacion = generar_explicacion(resultado, nombre, info)
        except Exception as e:
            log(f"Error Groq generar_explicacion: {e}", "ERR")
            status.update(label=":material/error: Error al contactar con Groq", state="error")
            st.error(f"No se pudo conectar con el agente IA (Groq). Comprueba tu GROQ_API_KEY y la conexión a Internet.\n\n`{e}`")
            st.stop()
        st.write(f"  Respuesta recibida: {len(explicacion)} caracteres")
        log(f"Respuesta recibida: {len(explicacion)} caracteres", "OK")

        status.update(label=":material/check_circle: Análisis generado correctamente", state="complete")

    st.session_state["explicacion_inicial"] = explicacion
    st.session_state["historial_chat"] = [{"role": "assistant", "content": explicacion}]

# ---------------------------------------------------------------------------
# Análisis inicial
# ---------------------------------------------------------------------------
section_label("ANÁLISIS INICIAL", icon="article")
with st.container(border=True):
    st.markdown(st.session_state["explicacion_inicial"])

divider()

# ---------------------------------------------------------------------------
# Chat de seguimiento + acciones laterales
# ---------------------------------------------------------------------------
col_chat, col_side = st.columns([1.8, 1])

with col_chat:
    section_label("CONVERSA CON EL AGENTE", icon="forum")
    for msg in st.session_state["historial_chat"][1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pregunta := st.chat_input("Escribe tu pregunta…  p. ej. ¿qué le falta para tier máximo?"):
        log(f"Usuario pregunta: {pregunta[:80]}{'…' if len(pregunta) > 80 else ''}", "STEP")
        st.session_state["historial_chat"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.status(":material/bolt: Consultando Groq…", expanded=False) as s:
                st.write(f"  Turno nº {len(st.session_state['historial_chat'])}")
                st.write(f"  Pregunta: *{pregunta[:120]}*")
                log(f"Llamando a Groq API — chat() turno {len(st.session_state['historial_chat'])}", "API")
                try:
                    respuesta = chat(
                        historial=st.session_state["historial_chat"][:-1],
                        pregunta=pregunta,
                        resultado=resultado,
                        nombre=nombre,
                        info_adicional=info,
                    )
                except Exception as e:
                    log(f"Error Groq chat: {e}", "ERR")
                    s.update(label=":material/error: Error al contactar con Groq", state="error")
                    st.error(f"No se pudo obtener respuesta del agente IA.\n\n`{e}`")
                    st.session_state["historial_chat"].pop()
                    st.stop()
                st.write(f"  Respuesta recibida: {len(respuesta)} caracteres")
                log(f"Respuesta chat recibida: {len(respuesta)} chars", "OK")
                s.update(label=":material/check_circle: Respuesta generada", state="complete")

            st.markdown(respuesta)

        st.session_state["historial_chat"].append({"role": "assistant", "content": respuesta})

with col_side:
    section_label("PREGUNTA ALGO", icon="lightbulb")

    # Sugerencias adaptadas al contexto de la predicción
    _siguiente_tier = {"bajo": "MEDIO", "medio": "ALTO", "alto": "el tier máximo ya"}
    _tier_ref = {"bajo": "underground del dataset", "medio": "de tier BAJO para comparar", "alto": "de tier MEDIO para comparar"}
    _confianza_baja = proba[nivel] < 0.55

    sugerencias = [
        f"¿Qué métricas concretas le faltan a {nombre} para llegar a {_siguiente_tier[nivel]}?",
        ("¿Por qué el modelo tiene tan poca confianza en esta predicción?"
         if _confianza_baja else
         f"¿Qué métrica es la más determinante para que {nombre} esté en {nivel.upper()}?"),
        f"¿Cómo se compara {nombre} con un artista {_tier_ref[nivel]}?",
        "¿Cómo se entrenó el modelo y qué significa la confianza?",
    ]

    for i, sug in enumerate(sugerencias):
        if st.button(sug, key=f"sug_{i}", use_container_width=True, type="secondary"):
            st.session_state["historial_chat"].append({"role": "user", "content": sug})
            with st.spinner(":material/bolt: Consultando Groq…"):
                try:
                    respuesta = chat(
                        historial=st.session_state["historial_chat"][:-1],
                        pregunta=sug,
                        resultado=resultado,
                        nombre=nombre,
                        info_adicional=info,
                    )
                except Exception as e:
                    log(f"Error Groq sugerencia: {e}", "ERR")
                    st.session_state["historial_chat"].pop()
                    st.error(f"No se pudo obtener respuesta del agente IA.\n\n`{e}`")
                    st.stop()
            st.session_state["historial_chat"].append({"role": "assistant", "content": respuesta})
            st.rerun()

    st.markdown("&nbsp;", unsafe_allow_html=True)
    section_label("ACCIONES", icon="settings")
    if len(st.session_state["historial_chat"]) > 1:
        if st.button(":material/delete_sweep: Limpiar conversación", use_container_width=True, type="secondary"):
            st.session_state["historial_chat"] = [
                {"role": "assistant", "content": st.session_state["explicacion_inicial"]}
            ]
            log("Conversación limpiada por el usuario", "INFO")
            st.rerun()
    if st.button(":material/restart_alt: Nueva predicción", use_container_width=True):
        reset_prediccion()
        log("Nueva predicción iniciada desde Análisis IA", "INFO")
        st.switch_page("pages/2_Prediccion.py")
