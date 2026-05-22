"""
Página 2 — Predice un artista.

Flujo principal (automático):
  1. El usuario escribe el nombre → búsqueda en Spotify
  2. Confirma el artista correcto (o auto-confirma si score ≥ 0.90)
  3. Las APIs se llaman en paralelo automáticamente
  4. Se muestran los datos encontrados + botón de predicción

Modo manual (expander): formulario original intacto para uso sin APIs o corrección.
"""

import os
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.models.predict import construir_features, predecir, _cargar_modelo
from src.models.shap_explainer import shap_waterfall_fig
from src.utils.log_streamlit import log, render_sidebar_log, reset_prediccion
from src.utils.feature_labels import (
    GRUPOS_DISPLAY, FEATURE_LABELS, format_valor, es_tecnica, detectar_alertas,
)
from src.data_collectors.fetch_artista import (
    CandidatoSpotify, ResultadoFetch,
    buscar_candidatos_spotify, desambiguar_con_groq,
    fetch_features_por_nombre, resultado_a_features,
)

from styles import (
    inject_styles, page_header, brand_block, section_label, divider,
    result_disc, TIER_COLOR, TIER_DESC, PALETTE,
)

st.set_page_config(page_title="Predicción · TFM", page_icon="🎤", layout="wide")
inject_styles()

with st.sidebar:
    st.markdown(brand_block(), unsafe_allow_html=True)
    render_sidebar_log()

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    page_header(
        crumb="Inicio · <b style='color:#B6ADCB;'>Predicción</b>",
        title_html="Predice un <em>artista</em>",
        sub="Escribe el nombre del artista. Los datos se obtienen automáticamente de Spotify, Last.fm, YouTube y setlist.fm.",
        pill_text="Live",
        pill_color="mint",
    ),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Clientes API cacheados (se inicializan una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_sp_client():
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))


# ---------------------------------------------------------------------------
# Inicialización de estado
# ---------------------------------------------------------------------------

_ESTADO_INIT = {
    "fetch_estado":            "idle",
    "fetch_candidatos":        [],
    "fetch_candidato":         None,
    "fetch_resultado":         None,
    "fetch_nombre":            "",
    "fetch_features_override": {},
    "fetch_groq_sugerido":     None,
}
for k, v in _ESTADO_INIT.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helper: renderizar resultado de predicción (compartido entre ambos flujos)
# ---------------------------------------------------------------------------

def _render_resultado(nombre_clean: str, resultado: dict, info_conciertos: str = ""):
    features = resultado["features"]
    nivel    = resultado["nivel"]
    proba    = resultado["probabilidades"]

    color = TIER_COLOR[nivel]
    pct   = int(round(proba[nivel] * 100))

    if pct >= 70:
        conf_label, conf_color = "ALTA CONFIANZA",  PALETTE["mint"]
    elif pct >= 55:
        conf_label, conf_color = "CONFIANZA MEDIA", PALETTE["amber"]
    else:
        conf_label, conf_color = "BAJA CONFIANZA",  PALETTE["coral"]

    bars_html = ""
    for n in ("bajo", "medio", "alto"):
        c = TIER_COLOR[n]
        p = proba[n] * 100
        bars_html += f"""
        <div style="display:flex;align-items:center;gap:12px;background:rgba(0,0,0,0.25);
                    padding:10px 14px;border-radius:11px;border:1px solid {c}33;">
          <span style="font-family:'JetBrains Mono';font-size:11px;color:#B6ADCB;width:50px;text-transform:uppercase;">{n}</span>
          <div style="flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
            <div style="width:{p:.1f}%;height:100%;background:{c};border-radius:4px;"></div>
          </div>
          <span style="font-family:'JetBrains Mono';font-size:12px;color:{c};font-weight:600;min-width:48px;text-align:right;">{p:.0f}%</span>
        </div>"""

    st.markdown(
        f"""
        <div style="background:linear-gradient(140deg,{color}26,{color}08 80%);
                    border:1px solid {color}66;border-radius:22px;padding:30px;
                    display:grid;grid-template-columns:auto 1fr 220px;gap:28px;align-items:center;
                    position:relative;overflow:hidden;margin-top:24px;">
          {result_disc(pct, color).strip()}
          <div>
            <div style="font-family:'JetBrains Mono';font-size:11px;color:{color};text-transform:uppercase;
                        letter-spacing:0.12em;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              <span style="font-family:'Material Symbols Rounded';font-size:13px;vertical-align:-2px;">verified</span>
              &nbsp;PREDICCIÓN COMPLETADA
              <span style="background:{conf_color}22;color:{conf_color};border:1px solid {conf_color}55;
                           border-radius:6px;padding:2px 8px;font-size:10px;letter-spacing:0.1em;">{conf_label}</span>
            </div>
            <h3 style="font-family:'Space Grotesk';font-size:32px;margin:8px 0 6px;letter-spacing:-0.02em;">
              {nombre_clean} &nbsp;·&nbsp; tier <span style="color:{color};">{nivel.upper()}</span>
            </h3>
            <p style="margin:0;color:#B6ADCB;font-size:14px;">{TIER_DESC[nivel]}</p>
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            {bars_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alertas = detectar_alertas(features, nivel, proba)
    if alertas:
        section_label("AVISOS DEL MODELO", icon="bolt")
        for alerta in alertas:
            if alerta["tipo"] == "warning":
                st.warning(alerta["msg"], icon=":material/warning:")
            else:
                st.info(alerta["msg"], icon=":material/info:")

    divider()
    with st.expander(":material/database:  Ver todas las variables enviadas al modelo (32 variables)"):
        st.caption(
            "Cada sección muestra las variables de una fuente de datos. "
            "Las marcadas como *(calculado)* o *(interno)* son transformaciones matemáticas."
        )
        for nombre_grupo, keys in GRUPOS_DISPLAY:
            st.markdown(f"**{nombre_grupo}**")
            rows_main, rows_tech = [], []
            for key in keys:
                if key not in features:
                    continue
                label = FEATURE_LABELS.get(key, key)
                valor = format_valor(key, features[key])
                (rows_tech if es_tecnica(key) else rows_main).append({"Variable": label, "Valor": valor})
            if rows_main:
                st.dataframe(pd.DataFrame(rows_main), hide_index=True, use_container_width=True)
            if rows_tech:
                with st.expander(f"Variables internas ({len(rows_tech)})", expanded=False):
                    st.dataframe(pd.DataFrame(rows_tech), hide_index=True, use_container_width=True)
            st.markdown("")

    divider()
    section_label("¿POR QUÉ ESTA PREDICCIÓN?  ·  EXPLICACIÓN SHAP", icon="find_in_page")
    st.caption(
        "Cada barra muestra cuánto empuja una variable la predicción hacia el tier predicho. "
        "Verde = empuja hacia arriba · Rosa = empuja hacia abajo."
    )
    try:
        with st.spinner("Calculando valores SHAP…"):
            shap_fig, _ = shap_waterfall_fig(features)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)
    except FileNotFoundError:
        st.info("Modelo no disponible. Ejecuta `python -m src.models.train`.", icon=":material/info:")

    divider()
    st.success(
        "Predicción guardada. Ve a **Análisis IA** para que el agente te explique el resultado.",
        icon=":material/auto_awesome:",
    )
    c_new, _ = st.columns([1, 3])
    with c_new:
        if st.button(":material/refresh: Nueva predicción", use_container_width=True, type="secondary"):
            st.session_state["prediccion"] = None
            for k, v in _ESTADO_INIT.items():
                st.session_state[k] = v
            st.rerun()


# ---------------------------------------------------------------------------
# Si ya hay predicción guardada → mostrarla y parar
# ---------------------------------------------------------------------------

if st.session_state.get("prediccion"):
    pred = st.session_state["prediccion"]
    _render_resultado(pred["nombre"], pred["resultado"], pred.get("info_conciertos", ""))
    st.stop()


# ---------------------------------------------------------------------------
# Flujo automático — máquina de estados
# ---------------------------------------------------------------------------

estado = st.session_state["fetch_estado"]

# ── idle: input de nombre ──────────────────────────────────────────────────
if estado == "idle":
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        nombre_input = st.text_input(
            "Nombre del artista",
            placeholder="Ej: Rels B, Recycled J, La Zowi, Natos y Waor…",
            label_visibility="collapsed",
        )
    with col_btn:
        buscar = st.button(":material/search: Buscar", use_container_width=True)

    if buscar and nombre_input.strip():
        nombre_q = nombre_input.strip()
        with st.spinner(f"Buscando «{nombre_q}» en Spotify…"):
            try:
                sp = _get_sp_client()
                candidatos = buscar_candidatos_spotify(nombre_q, sp, limit=5)
            except Exception as e:
                st.error(f"No se pudo conectar con Spotify: {e}. Usa el modo manual.")
                candidatos = []

        if not candidatos:
            st.warning(
                f"No se encontró ningún artista para «{nombre_q}» en Spotify. "
                "Prueba con el modo manual.",
                icon=":material/search_off:",
            )
        elif candidatos[0].score >= 0.90 and (
            len(candidatos) < 2 or candidatos[0].score - candidatos[1].score > 0.05
        ):
            # Auto-confirmar: coincidencia clara
            st.session_state["fetch_candidato"] = candidatos[0]
            st.session_state["fetch_nombre"]    = nombre_q
            st.session_state["fetch_estado"]    = "confirmado"
            st.rerun()
        else:
            # Necesita confirmación del usuario
            groq_sugerido = None
            if (
                len(candidatos) >= 2
                and candidatos[0].score - candidatos[1].score <= 0.05
                and os.getenv("GROQ_API_KEY")
            ):
                try:
                    groq_sugerido = desambiguar_con_groq(nombre_q, candidatos, os.getenv("GROQ_API_KEY"))
                except Exception:
                    pass
            st.session_state["fetch_candidatos"]    = candidatos
            st.session_state["fetch_nombre"]        = nombre_q
            st.session_state["fetch_groq_sugerido"] = groq_sugerido
            st.session_state["fetch_estado"]        = "candidatos_listos"
            st.rerun()

# ── candidatos_listos: selector ────────────────────────────────────────────
elif estado == "candidatos_listos":
    candidatos: list = st.session_state["fetch_candidatos"]
    nombre_q: str    = st.session_state["fetch_nombre"]
    groq_sug         = st.session_state.get("fetch_groq_sugerido")

    st.markdown(
        f'<p style="color:#B6ADCB;margin-bottom:12px;">Resultados para <b>«{nombre_q}»</b>. '
        'Selecciona el artista correcto:</p>',
        unsafe_allow_html=True,
    )

    opciones = []
    for c in candidatos:
        generos_txt = ", ".join(c.generos[:3]) if c.generos else "sin géneros"
        ia_badge = " ✦ IA recomienda" if groq_sug and groq_sug.spotify_id == c.spotify_id else ""
        opciones.append(f"{c.nombre_spotify}  —  {generos_txt}  ·  {c.score:.0%}{ia_badge}")

    idx_default = 0
    if groq_sug:
        ids = [c.spotify_id for c in candidatos]
        if groq_sug.spotify_id in ids:
            idx_default = ids.index(groq_sug.spotify_id)

    seleccion = st.radio(
        "Artista",
        opciones,
        index=idx_default,
        label_visibility="collapsed",
    )
    idx_sel = opciones.index(seleccion)

    # Mostrar imagen del candidato seleccionado
    candidato_sel = candidatos[idx_sel]
    if candidato_sel.imagen_url:
        col_img, col_info = st.columns([1, 4])
        with col_img:
            st.image(candidato_sel.imagen_url, width=80)
        with col_info:
            st.markdown(
                f"**{candidato_sel.nombre_spotify}** · score {candidato_sel.score:.0%}  \n"
                f"`ID: {candidato_sel.spotify_id}`"
            )
    else:
        st.caption(f"Spotify ID: {candidato_sel.spotify_id}  ·  score {candidato_sel.score:.0%}")

    c_conf, c_back = st.columns([2, 1])
    with c_conf:
        if st.button(":material/check: Confirmar este artista", use_container_width=True):
            st.session_state["fetch_candidato"] = candidato_sel
            st.session_state["fetch_estado"]    = "confirmado"
            st.rerun()
    with c_back:
        if st.button(":material/arrow_back: Volver a buscar", use_container_width=True, type="secondary"):
            st.session_state["fetch_estado"] = "idle"
            st.rerun()

def _resumen_plataforma(plataforma: str, datos: dict) -> str:
    if plataforma == "spotify":
        return (
            f"{datos.get('sp_num_albums', 0)} álbumes · "
            f"{datos.get('sp_num_singles', 0)} singles · "
            f"{float(datos.get('sp_anos_activo', 0)):.0f} años activo"
        )
    if plataforma == "lastfm":
        return (
            f"{int(datos.get('lfm_oyentes', 0)):,} oyentes · "
            f"{int(datos.get('lfm_scrobbles', 0)):,} scrobbles"
        )
    if plataforma == "youtube":
        return (
            f"{int(datos.get('yt_suscriptores', 0)):,} suscriptores · "
            f"{int(datos.get('yt_vistas_totales', 0)):,} vistas"
        )
    if plataforma == "setlistfm":
        return (
            f"{int(datos.get('sl_num_conciertos', 0))} conciertos · "
            f"{int(datos.get('sl_num_paises', 0))} países"
        )
    if plataforma == "tendencias":
        g = datos.get("trend_gtrends_interes_medio", 0)
        return f"Google Trends: {g:.0f}/100"
    return ""


# ── confirmado: llamar a las APIs y transitar a fetch_completo ─────────────
if estado == "confirmado":
    candidato: CandidatoSpotify = st.session_state["fetch_candidato"]

    _ICONOS_EST = {
        "ok":        ":material/check_circle:",
        "not_found": ":material/cancel:",
        "error":     ":material/warning:",
        "skipped":   ":material/remove:",
    }

    with st.status(
        f":material/bolt: Obteniendo datos de **{candidato.nombre_spotify}**…",
        expanded=True,
    ) as status_widget:

        def _on_progress(plataforma, est, datos):
            icono = _ICONOS_EST.get(est, ":material/remove:")
            if est == "ok":
                resumen = _resumen_plataforma(plataforma, datos)
                st.write(f"{icono} **{plataforma.capitalize()}** — {resumen}")
            elif est == "not_found":
                st.write(f"{icono} **{plataforma.capitalize()}** — no encontrado")
            elif est == "error":
                st.write(f"{icono} **{plataforma.capitalize()}** — error de conexión")
            elif est == "skipped":
                st.write(f"{icono} **{plataforma.capitalize()}** — API key no configurada")

        resultado_fetch = fetch_features_por_nombre(
            st.session_state["fetch_nombre"],
            candidato.spotify_id,
            candidato.nombre_spotify,
            sp_client=_get_sp_client(),
            on_progress=_on_progress,
        )
        status_widget.update(label=":material/check_circle: Datos obtenidos", state="complete")

    st.session_state["fetch_resultado"] = resultado_fetch
    st.session_state["fetch_estado"]    = "fetch_completo"
    estado = "fetch_completo"


# ── fetch_completo: mostrar tabla de resultados + botón de predicción ─────
if estado == "fetch_completo":
    resultado_fetch: ResultadoFetch = st.session_state["fetch_resultado"]
    nombre_clean = resultado_fetch.nombre_spotify

    # Advertencias de la API
    for aviso in resultado_fetch.advertencias:
        st.warning(aviso, icon=":material/warning:")

    # Tabla resumen por plataforma
    section_label("Datos obtenidos", icon="database")
    _ICONOS_TABLA = {"ok": "✓", "not_found": "✗", "error": "⚠", "skipped": "—"}
    _COLS_TABLA = {
        "spotify":    ("Spotify",    ["sp_num_albums", "sp_num_singles", "sp_anos_activo"]),
        "lastfm":     ("Last.fm",    ["lfm_oyentes", "lfm_scrobbles"]),
        "youtube":    ("YouTube",    ["yt_suscriptores", "yt_vistas_totales"]),
        "setlistfm":  ("setlist.fm", ["sl_num_conciertos", "sl_num_paises"]),
        "tendencias": ("Tendencias", ["trend_gtrends_interes_medio", "trend_yt_vistas_recientes"]),
    }
    _ATTR_MAP = {
        "spotify": "sp", "lastfm": "lfm", "youtube": "yt",
        "setlistfm": "sl", "tendencias": "trend",
    }

    for plataforma, (label, campos) in _COLS_TABLA.items():
        est = resultado_fetch.estado.get(plataforma, "skipped")
        icono = _ICONOS_TABLA[est]
        datos_plat = getattr(resultado_fetch, _ATTR_MAP[plataforma], {})

        valores = []
        for campo in campos:
            if campo in datos_plat:
                val = datos_plat[campo]
                label_campo = FEATURE_LABELS.get(campo, campo)
                if isinstance(val, float) and val > 1000:
                    valores.append(f"{int(val):,}")
                elif isinstance(val, float):
                    valores.append(f"{val:.1f}")
                else:
                    valores.append(str(val))
        valores_txt = " · ".join(valores) if valores else "—"

        color_est = PALETTE["mint"] if est == "ok" else PALETTE["coral"] if est == "not_found" else PALETTE["amber"]
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:14px;
                        background:var(--surface);border:1px solid var(--line);
                        border-radius:12px;padding:10px 16px;margin-bottom:6px;">
              <span style="font-family:'JetBrains Mono';font-size:16px;color:{color_est};
                           width:20px;text-align:center;">{icono}</span>
              <span style="font-weight:600;color:#F2EEFA;min-width:90px;">{label}</span>
              <span style="color:#B6ADCB;font-size:13px;">{valores_txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Corrección manual inline (solo campos clave no encontrados)
    _mostrar_override = any(
        resultado_fetch.estado.get(p) in ("not_found", "error")
        for p in ("lastfm", "youtube", "setlistfm")
    )
    overrides = {}
    if _mostrar_override:
        with st.expander(":material/edit: Corregir datos no encontrados", expanded=False):
            st.caption(
                "Introduce los valores manualmente para las plataformas no encontradas. "
                "La feature más importante del modelo es el número de conciertos."
            )
            cols_ov = st.columns(3)
            if resultado_fetch.estado.get("lastfm") in ("not_found", "error"):
                with cols_ov[0]:
                    overrides["lfm_oyentes"]   = st.number_input("Oyentes Last.fm",   min_value=0, value=0, step=1000)
                    overrides["lfm_scrobbles"] = st.number_input("Scrobbles Last.fm", min_value=0, value=0, step=10000)
            if resultado_fetch.estado.get("youtube") in ("not_found", "error"):
                with cols_ov[1]:
                    overrides["yt_suscriptores"]   = st.number_input("Suscriptores YouTube",  min_value=0, value=0, step=1000)
                    overrides["yt_vistas_totales"]  = st.number_input("Vistas YouTube",        min_value=0, value=0, step=10000)
            if resultado_fetch.estado.get("setlistfm") in ("not_found", "error"):
                with cols_ov[2]:
                    overrides["sl_num_conciertos"] = st.number_input("Nº conciertos (setlist.fm)", min_value=0, value=0, step=1)
                    overrides["sl_tiene_datos"]    = int(st.checkbox("Aparece en setlist.fm", value=False))

    st.session_state["fetch_features_override"] = overrides

    divider()
    col_pred, col_nuevo = st.columns([2, 1])
    with col_pred:
        predecir_btn = st.button(
            ":material/play_arrow: Predecir tier de sala",
            use_container_width=True,
        )
    with col_nuevo:
        if st.button(":material/refresh: Nueva búsqueda", use_container_width=True, type="secondary"):
            for k, v in _ESTADO_INIT.items():
                st.session_state[k] = v
            st.rerun()

    if predecir_btn:
        features_kwargs = resultado_a_features(resultado_fetch)
        features_kwargs.update(st.session_state.get("fetch_features_override", {}))

        log(f"Predicción automática para: {nombre_clean}", "STEP")
        resultado_pred = predecir(**features_kwargs)
        nivel = resultado_pred["nivel"]
        log(f"Predicción: {nivel.upper()}", "OK")

        st.session_state["prediccion"] = {
            "resultado":       resultado_pred,
            "nombre":          nombre_clean,
            "info_conciertos": "",
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Modo manual — formulario original en expander
# ---------------------------------------------------------------------------

divider()

def _src_header(icon: str, title: str, color_hex: str, meta: str = "", url: str = ""):
    if url:
        meta_html = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer"'
            f' style="margin-left:auto;font-family:\'JetBrains Mono\';font-size:10.5px;'
            f'color:#7A7290;text-transform:uppercase;letter-spacing:0.08em;'
            f'text-decoration:none;display:inline-flex;align-items:center;gap:5px;">'
            f'{meta}'
            f'<span style="font-family:\'Material Symbols Rounded\';font-size:13px;vertical-align:-2px;">open_in_new</span>'
            f'</a>'
        )
    else:
        meta_html = (
            f'<span style="margin-left:auto;font-family:\'JetBrains Mono\';font-size:10.5px;'
            f'color:#7A7290;text-transform:uppercase;letter-spacing:0.08em;">{meta}</span>'
        )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:18px 0 10px;">
          <div style="width:34px;height:34px;border-radius:10px;
                      background:{color_hex}26;color:{color_hex};
                      display:flex;align-items:center;justify-content:center;
                      font-family:'Material Symbols Rounded';font-size:18px;">
            {icon}
          </div>
          <h4 style="font-family:'Space Grotesk';font-size:16px;margin:0;">{title}</h4>
          {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.expander(":material/tune: Introducir datos manualmente (modo avanzado)", expanded=False):
    st.caption(
        "Usa este modo si la búsqueda automática no encuentra el artista "
        "o si quieres introducir los datos directamente."
    )

    with st.form("form_prediccion"):
        nombre = st.text_input(
            "Nombre del artista",
            placeholder="Ej: Rels B, Recycled J, La Zowi…",
        )

        _src_header("music_note", "Spotify", PALETTE["mint"], "3 campos · Spotify", "https://open.spotify.com")
        col_sp1, col_sp2, col_sp3 = st.columns(3)
        with col_sp1:
            sp_num_albums  = st.number_input("Nº álbumes", min_value=0, value=0, step=1)
        with col_sp2:
            sp_num_singles = st.number_input("Nº singles", min_value=0, value=0, step=1)
        with col_sp3:
            sp_anos_activo = st.number_input("Años activo", min_value=0.0, value=1.0, step=0.5, format="%.1f")

        _src_header("radio", "Last.fm", PALETTE["pink"], "2 campos · Last.fm", "https://www.last.fm")
        col_lfm1, col_lfm2 = st.columns(2)
        with col_lfm1:
            lfm_oyentes   = st.number_input("Oyentes únicos",    min_value=0, value=0, step=100)
        with col_lfm2:
            lfm_scrobbles = st.number_input("Scrobbles totales", min_value=0, value=0, step=1000)

        _src_header("smart_display", "YouTube", PALETTE["coral"], "3 campos · YouTube", "https://www.youtube.com")
        col_yt1, col_yt2, col_yt3 = st.columns(3)
        with col_yt1:
            yt_suscriptores   = st.number_input("Suscriptores",   min_value=0, value=0, step=100)
        with col_yt2:
            yt_vistas_totales = st.number_input("Vistas totales", min_value=0, value=0, step=1000)
        with col_yt3:
            yt_num_videos     = st.number_input("Nº vídeos", min_value=0, value=0, step=1)

        _src_header("stadium", "Conciertos (setlist.fm)", PALETTE["blue"], "2 campos · Setlist.fm", "https://www.setlist.fm")
        col_sl1, col_sl2 = st.columns([1, 1])
        with col_sl1:
            sl_tiene_datos = st.checkbox(
                "El artista aparece en setlist.fm",
                value=False,
            )
            sl_num_conciertos = st.number_input(
                "Nº conciertos documentados",
                min_value=0, value=0, step=1,
            )
        with col_sl2:
            info_conciertos = st.text_area(
                "Info adicional sobre directos (opcional)",
                placeholder="Ej: Tocó en Sala Copera con ~150 personas. Ha actuado en festivales locales.",
                help="No entra en el modelo — el agente IA la usa para contextualizar.",
                height=130,
            )

        submitted = st.form_submit_button(
            ":material/play_arrow: Predecir tier de sala",
            use_container_width=True,
        )

    if submitted:
        nombre_clean = nombre.strip() if nombre else "Artista"
        log(f"Nueva predicción manual para: {nombre_clean}", "STEP")

        with st.status(f":material/bolt: Analizando {nombre_clean}…", expanded=True) as status:
            inputs_raw = {
                "sp_num_albums":     int(sp_num_albums),
                "sp_num_singles":    int(sp_num_singles),
                "sp_anos_activo":    float(sp_anos_activo),
                "lfm_oyentes":       int(lfm_oyentes),
                "lfm_scrobbles":     int(lfm_scrobbles),
                "yt_suscriptores":   int(yt_suscriptores),
                "yt_vistas_totales": int(yt_vistas_totales),
                "yt_num_videos":     int(yt_num_videos),
                "sl_num_conciertos": int(sl_num_conciertos),
                "sl_tiene_datos":    1 if int(sl_num_conciertos) > 0 else int(sl_tiene_datos),
            }
            st.write("**1 · Construyendo vector de features…**")
            features = construir_features(**inputs_raw)
            st.write(f"  Lanzamientos/año: **{features['sp_releases_por_ano']:.1f}**")
            st.write(f"  Fidelidad (scrobbles/oyente): **{features['lfm_scrobbles_por_oyente']:.1f}**")

            st.write("**2 · Ejecutando la predicción…**")
            resultado = predecir(**inputs_raw)
            nivel = resultado["nivel"]
            proba = resultado["probabilidades"]
            st.write(f"  → **Tier predicho: {nivel.upper()}**")
            log(f"Predicción: {nivel.upper()}", "OK")
            status.update(label=f":material/check_circle: {nivel.upper()}", state="complete")

        st.session_state["prediccion"] = {
            "resultado":       resultado,
            "nombre":          nombre_clean,
            "info_conciertos": info_conciertos.strip(),
        }
        st.rerun()
