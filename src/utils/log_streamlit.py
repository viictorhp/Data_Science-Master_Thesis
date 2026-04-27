"""
log_streamlit.py
================
Utilidad de logging para el dashboard Streamlit.
Escribe en st.session_state["session_log"] y renderiza
un panel persistente en la barra lateral.
"""

from datetime import datetime
import streamlit as st

_ICONS = {
    "INFO":  "ℹ️",
    "OK":    "✅",
    "WARN":  "⚠️",
    "ERR":   "❌",
    "ML":    "🤖",
    "API":   "🌐",
    "DATA":  "📊",
    "STEP":  "▶️",
}


def log(msg: str, level: str = "INFO") -> None:
    """Añade una entrada al log de sesión."""
    if "session_log" not in st.session_state:
        st.session_state["session_log"] = []
    ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icon = _ICONS.get(level, "•")
    st.session_state["session_log"].append(f"{ts}  {icon}  {msg}")


# Claves de session_state asociadas a una predicción concreta.
# Se borran todas al iniciar una nueva búsqueda.
_PREDICTION_KEYS = [
    "prediccion",
    "explicacion_inicial",
    "historial_chat",
    "prediccion_id",
    "prediccion_hash",
]


def reset_prediccion() -> None:
    """Borra del session_state todo lo relacionado con la predicción activa."""
    for key in _PREDICTION_KEYS:
        st.session_state.pop(key, None)
    log("Session reseteada — nueva predicción lista", "INFO")


def render_sidebar_log() -> None:
    """Renderiza el panel de log y el botón de reset en la barra lateral."""
    with st.sidebar:

        # Botón de nueva búsqueda — siempre visible
        if st.session_state.get("prediccion"):
            nombre = st.session_state["prediccion"].get("nombre", "artista")
            st.info(f"Predicción activa: **{nombre}**")
        if st.button("🔄 Nueva predicción", key="btn_reset", width='stretch'):
            reset_prediccion()
            st.rerun()

        st.divider()

        # Log de sesión
        entries = st.session_state.get("session_log", [])
        st.markdown(f"**📋 Log de sesión** `{len(entries)} eventos`")
        if entries:
            st.text_area(
                label="log",
                value="\n".join(reversed(entries[-80:])),
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            if st.button("🗑️ Limpiar log", key="btn_clear_log"):
                st.session_state["session_log"] = []
                st.rerun()
        else:
            st.caption("Sin actividad todavía.")
