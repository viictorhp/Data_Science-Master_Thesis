import sys
sys.path.insert(0, '.')

import streamlit as st

from src.agents.explicador import chat, generar_explicacion, _system_prompt
from src.utils.log_streamlit import log, render_sidebar_log, reset_prediccion

st.set_page_config(
    page_title="Análisis IA · TFM",
    page_icon="🤖",
    layout="wide",
)

render_sidebar_log()

st.title("🤖 Análisis con IA")
st.caption("Agente LangChain + Groq (Llama 3.3 70B) · explica la predicción y responde preguntas.")

# ---------------------------------------------------------------------------
# Guardia: necesita predicción previa
# ---------------------------------------------------------------------------
if "prediccion" not in st.session_state:
    st.warning("⚠️ Aún no hay ninguna predicción. Ve primero a **🎤 Prediccion**, rellena el formulario y pulsa *Predecir*.")
    st.stop()

pred      = st.session_state["prediccion"]
resultado = pred["resultado"]
nombre    = pred["nombre"]
info      = pred["info_conciertos"]
nivel     = resultado["nivel"]
proba     = resultado["probabilidades"]
features  = resultado["features"]

# ---------------------------------------------------------------------------
# Resumen de la predicción activa
# ---------------------------------------------------------------------------
color = {"bajo": "🔴", "medio": "🟡", "alto": "🟢"}[nivel]
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Artista:** {nombre}")
    c2.markdown(f"**Tier:** {color} {nivel.upper()}")
    c3.metric("Confianza", f"{proba[nivel]:.1%}")
    segundo = sorted(proba.items(), key=lambda x: x[1], reverse=True)[1]
    c4.metric("2º más probable", f"{segundo[0]}: {segundo[1]:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Explicación inicial (se genera una sola vez por predicción)
# ---------------------------------------------------------------------------
pred_id = id(resultado)
if "explicacion_inicial" not in st.session_state or \
   st.session_state.get("prediccion_id") != pred_id:

    st.session_state["prediccion_id"] = pred_id

    with st.status("🌐 Consultando agente IA...", expanded=True) as status:

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
        st.write(f"    • Directo  : {features['sl_num_conciertos']:.0f} conciertos · setlist.fm: {'✅ sí' if tiene_sl else '❌ no aparece'}")
        st.write(f"  Info adicional: {'Sí' if info else 'No'}")
        st.write(f"  Longitud del system prompt: {len(system_prompt)} caracteres")
        log(f"System prompt construido para {nombre}: {len(system_prompt)} chars", "API")

        with st.expander("📄 Ver system prompt completo enviado a la IA"):
            st.code(system_prompt, language=None)

        st.write("**2 · Enviando petición a la API de Groq**")
        st.write("  ⏳ Esperando respuesta...")
        log("Llamando a Groq API — generar_explicacion()", "API")

        explicacion = generar_explicacion(resultado, nombre, info)

        st.write(f"  ✅ Respuesta recibida: {len(explicacion)} caracteres")
        log(f"Respuesta recibida: {len(explicacion)} caracteres", "OK")

        status.update(label="✅ Análisis generado correctamente", state="complete")

    st.session_state["explicacion_inicial"] = explicacion
    st.session_state["historial_chat"] = [
        {"role": "assistant", "content": explicacion}
    ]

st.markdown("### 📝 Análisis inicial")
st.markdown(st.session_state["explicacion_inicial"])

st.divider()

# ---------------------------------------------------------------------------
# Chat de seguimiento
# ---------------------------------------------------------------------------
st.markdown("### 💬 Pregunta al agente")
st.caption("Puedes preguntar qué mejorar, pedir más detalle, comparar con otros artistas...")

for msg in st.session_state["historial_chat"][1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pregunta := st.chat_input("Escribe tu pregunta aquí..."):
    log(f"Usuario pregunta: {pregunta[:80]}{'...' if len(pregunta) > 80 else ''}", "STEP")
    st.session_state["historial_chat"].append({"role": "user", "content": pregunta})

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.status("🌐 Consultando Groq...", expanded=True) as s:
            st.write(f"  Turno nº {len(st.session_state['historial_chat'])} de la conversación")
            st.write(f"  Mensajes en historial: {len(st.session_state['historial_chat'])}")
            st.write(f"  Pregunta: *{pregunta[:120]}*")
            log(f"Llamando a Groq API — chat() turno {len(st.session_state['historial_chat'])}", "API")

            respuesta = chat(
                historial=st.session_state["historial_chat"][:-1],
                pregunta=pregunta,
                resultado=resultado,
                nombre=nombre,
                info_adicional=info,
            )
            st.write(f"  ✅ Respuesta recibida: {len(respuesta)} caracteres")
            log(f"Respuesta chat recibida: {len(respuesta)} chars", "OK")
            s.update(label="✅ Respuesta generada", state="complete")

        st.markdown(respuesta)

    st.session_state["historial_chat"].append({"role": "assistant", "content": respuesta})

# ---------------------------------------------------------------------------
# Botón para limpiar el chat
# ---------------------------------------------------------------------------
col_btn1, col_btn2 = st.columns(2)
if len(st.session_state["historial_chat"]) > 1:
    with col_btn1:
        if st.button("🗑️ Limpiar conversación", type="secondary", width='stretch'):
            st.session_state["historial_chat"] = [
                {"role": "assistant", "content": st.session_state["explicacion_inicial"]}
            ]
            log("Conversación limpiada por el usuario", "INFO")
            st.rerun()
with col_btn2:
    if st.button("🔄 Nueva predicción", type="primary", width='stretch'):
        reset_prediccion()
        log("Nueva predicción iniciada desde Análisis IA", "INFO")
        st.switch_page("pages/2_Prediccion.py")
