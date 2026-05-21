"""
Página 2 — Predice un artista.

Misma lógica que la versión original (src.models.predict, shap_explainer, log_streamlit)
con el sistema de diseño Studio aplicado.
"""

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
        sub=("Rellena lo que sepas. Todos los campos son opcionales — el modelo asume "
             "valores neutros para lo que dejes en blanco."),
        pill_text="Live",
        pill_color="mint",
    ),
    unsafe_allow_html=True,
)


def _src_header(icon: str, title: str, color_hex: str, meta: str = "", url: str = ""):
    """Cabecera coloreada por fuente de datos dentro del formulario."""
    if url:
        meta_html = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer"'
            f' style="margin-left:auto;font-family:\'JetBrains Mono\';font-size:10.5px;'
            f'color:#7A7290;text-transform:uppercase;letter-spacing:0.08em;'
            f'text-decoration:none;display:inline-flex;align-items:center;gap:5px;'
            f'transition:color 0.15s;"'
            f' onmouseover="this.style.color=\'{color_hex}\'"'
            f' onmouseout="this.style.color=\'#7A7290\'">'
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


# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------
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
        yt_num_videos     = st.number_input(
            "Nº vídeos", min_value=0, value=0, step=1,
            help="Permite calcular las vistas medias por vídeo. Déjalo en 0 si no lo sabes.",
        )

    _src_header("stadium", "Conciertos (setlist.fm)", PALETTE["blue"], "2 campos · Setlist.fm", "https://www.setlist.fm")
    col_sl1, col_sl2 = st.columns([1, 1])
    with col_sl1:
        sl_tiene_datos = st.checkbox(
            "El artista aparece en setlist.fm",
            value=False,
            help="Marca si el artista tiene perfil en setlist.fm, aunque no se hayan encontrado conciertos. "
                 "Se activa automáticamente si introduces un número de conciertos > 0.",
        )
        sl_num_conciertos = st.number_input(
            "Nº conciertos documentados",
            min_value=0, value=0, step=1,
            help="Conciertos registrados. Si el artista no aparece en setlist.fm, deja 0 y desactiva el checkbox.",
        )
    with col_sl2:
        info_conciertos = st.text_area(
            "Info adicional sobre directos (opcional)",
            placeholder="Ej: Tocó en Sala Copera (Granada) con ~150 personas. Ha actuado en festivales locales.",
            help="No entra en el modelo — el agente IA la usa para contextualizar la explicación.",
            height=130,
        )

    submitted = st.form_submit_button(
        ":material/play_arrow: Predecir tier de sala",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Predicción con traza detallada
# ---------------------------------------------------------------------------
if submitted:
    nombre_clean = nombre.strip() if nombre else "Artista"
    log(f"Nueva predicción solicitada para: {nombre_clean}", "STEP")

    with st.status(f":material/bolt: Analizando {nombre_clean}…", expanded=True) as status:

        st.write("**1 · Inputs recibidos del usuario**")
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
        for k, v in inputs_raw.items():
            label = FEATURE_LABELS.get(k, k)
            st.write(f"  **{label}**: {v:,}")
        log(f"Inputs: {inputs_raw}", "DATA")

        st.write("**2 · Construyendo vector de 32 variables**")
        st.write("  Calculando ratios, logaritmos y rellenando variables no pedidas con valores neutros…")
        features = construir_features(**inputs_raw)
        log("Vector de features construido correctamente", "OK")
        st.write(f"  Lanzamientos/año: **{features['sp_releases_por_ano']:.1f}**")
        st.write(f"  Fidelidad (scrobbles/oyente): **{features['lfm_scrobbles_por_oyente']:.1f}**")
        st.write(f"  Vistas/suscriptor: **{features['yt_vistas_por_suscriptor']:.1f}**")
        st.write(f"  Vistas medias/vídeo: **{features['yt_vistas_por_video']:.0f}**")

        st.write("**3 · Cargando modelo**")
        model, meta = _cargar_modelo()
        st.write(f"  Entrenado con **{meta['n_samples']} artistas** · **{meta['n_features']} features**")
        st.write(f"  Fecha entrenamiento: {meta['trained_at'][:10]}")
        st.write(f"  Accuracy CV5: {meta['cv5_metrics']['acc_mean']:.1%} ± {meta['cv5_metrics']['acc_std']:.1%}")
        st.write(f"  F1 macro CV5: {meta['cv5_metrics']['f1_mean']:.1%} ± {meta['cv5_metrics']['f1_std']:.1%}")
        log(f"Modelo cargado: {meta['n_samples']} artistas, {meta['n_features']} features", "ML")

        st.write("**4 · Ejecutando la predicción**")
        resultado = predecir(**inputs_raw)
        nivel = resultado["nivel"]
        proba = resultado["probabilidades"]
        st.write(f"  Probabilidad BAJO  : {proba['bajo']:.1%}")
        st.write(f"  Probabilidad MEDIO : {proba['medio']:.1%}")
        st.write(f"  Probabilidad ALTO  : {proba['alto']:.1%}")
        st.write(f"  → **Tier predicho: {nivel.upper()}**")
        log(f"Predicción: {nivel.upper()} | bajo={proba['bajo']:.3f} medio={proba['medio']:.3f} alto={proba['alto']:.3f}", "OK")

        status.update(label=f":material/check_circle: Predicción completada → {nivel.upper()}", state="complete")

    st.session_state["prediccion"] = {
        "resultado":       resultado,
        "nombre":          nombre_clean,
        "info_conciertos": info_conciertos.strip(),
    }
    log(f"Predicción guardada en session_state para {nombre_clean}", "OK")

    # -------------------------------------------------------------------------
    # Resultado visual — disco + lista de tiers
    # -------------------------------------------------------------------------
    color = TIER_COLOR[nivel]
    pct = int(round(proba[nivel] * 100))

    if pct >= 70:
        conf_label, conf_color = "ALTA CONFIANZA", PALETTE["mint"]
    elif pct >= 55:
        conf_label, conf_color = "CONFIANZA MEDIA", PALETTE["amber"]
    else:
        conf_label, conf_color = "BAJA CONFIANZA", PALETTE["coral"]

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
        <div style="background:linear-gradient(140deg, {color}26, {color}08 80%);
                    border:1px solid {color}66;border-radius:22px;padding:30px;
                    display:grid;grid-template-columns:auto 1fr 220px;gap:28px;align-items:center;
                    position:relative;overflow:hidden;margin-top:24px;">
          {result_disc(pct, color)}
          <div>
            <div style="font-family:'JetBrains Mono';font-size:11px;color:{color};
                        text-transform:uppercase;letter-spacing:0.12em;
                        display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              <span style="font-family:'Material Symbols Rounded';font-size:13px;vertical-align:-2px;">verified</span>
              &nbsp;PREDICCIÓN COMPLETADA
              <span style="background:{conf_color}22;color:{conf_color};
                           border:1px solid {conf_color}55;border-radius:6px;
                           padding:2px 8px;font-size:10px;letter-spacing:0.1em;">
                {conf_label}
              </span>
            </div>
            <h3 style="font-family:'Space Grotesk';font-size:32px;margin:8px 0 6px;letter-spacing:-0.02em;">
              {nombre_clean} &nbsp; · &nbsp; tier <span style="color:{color};">{nivel.upper()}</span>
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

    # -------------------------------------------------------------------------
    # Alertas
    # -------------------------------------------------------------------------
    alertas = detectar_alertas(features, nivel, proba)
    if alertas:
        section_label("AVISOS DEL MODELO", icon="bolt")
        for alerta in alertas:
            if alerta["tipo"] == "warning":
                st.warning(alerta["msg"], icon=":material/warning:")
            else:
                st.info(alerta["msg"], icon=":material/info:")

    # -------------------------------------------------------------------------
    # Detalle de variables
    # -------------------------------------------------------------------------
    divider()
    with st.expander(":material/database:  Ver todas las variables enviadas al modelo (32 variables)"):
        st.caption(
            "Cada sección muestra las variables de una fuente de datos. "
            "Las marcadas como *(calculado)* o *(interno)* son transformaciones matemáticas "
            "que el modelo usa internamente — no las introduces tú directamente."
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
                with st.expander(f"Variables internas del modelo ({len(rows_tech)})", expanded=False):
                    st.dataframe(pd.DataFrame(rows_tech), hide_index=True, use_container_width=True)
            st.markdown("")

    # -------------------------------------------------------------------------
    # SHAP Waterfall
    # -------------------------------------------------------------------------
    divider()
    section_label("¿POR QUÉ ESTA PREDICCIÓN?  ·  EXPLICACIÓN SHAP", icon="find_in_page")
    st.caption(
        "Cada barra muestra cuánto empuja una variable la predicción hacia el tier predicho. "
        "Verde = empuja hacia arriba · Rosa = empuja hacia abajo. E[f(X)] = predicción media del modelo."
    )
    try:
        with st.spinner("Calculando valores SHAP…"):
            shap_fig, _ = shap_waterfall_fig(features)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)
        log("Waterfall SHAP generado correctamente", "ML")
    except FileNotFoundError:
        st.info(
            "El modelo serializado no está disponible. "
            "Ejecuta `python -m src.models.train` para entrenarlo.",
            icon=":material/info:",
        )
        log("Modelo no encontrado — SHAP no disponible", "WARN")

    divider()
    st.success(
        "Predicción guardada. Ve a **Análisis IA** para que el agente te explique el resultado.",
        icon=":material/auto_awesome:",
    )

    c_new, _ = st.columns([1, 3])
    with c_new:
        if st.button(":material/refresh: Nueva predicción", use_container_width=True, type="secondary"):
            reset_prediccion()
            st.rerun()
