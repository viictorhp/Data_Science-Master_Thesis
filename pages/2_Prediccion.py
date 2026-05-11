import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import streamlit as st

from src.models.predict import construir_features, predecir, _cargar_modelo
from src.models.shap_explainer import shap_waterfall_fig
from src.utils.log_streamlit import log, render_sidebar_log, reset_prediccion
from src.utils.feature_labels import (
    GRUPOS_DISPLAY, FEATURE_LABELS, format_valor, es_tecnica, detectar_alertas
)

st.set_page_config(
    page_title="Predice un artista · TFM",
    page_icon="🎤",
    layout="wide",
)

render_sidebar_log()

st.title("🎤 Predice un artista")
st.caption("Introduce las métricas básicas del artista — todos los campos son opcionales y empiezan en 0.")

# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------
with st.form("form_prediccion"):

    nombre = st.text_input("Nombre del artista", placeholder="Ej: Artista X")

    st.divider()
    col_sp, col_lfm = st.columns(2)

    with col_sp:
        st.markdown("**🎧 Spotify**")
        sp_num_albums  = st.number_input("Nº álbumes",  min_value=0, value=0, step=1)
        sp_num_singles = st.number_input("Nº singles",  min_value=0, value=0, step=1)
        sp_anos_activo = st.number_input("Años activo", min_value=0.0, value=1.0,
                                         step=0.5, format="%.1f")

    with col_lfm:
        st.markdown("**📻 Last.fm** — [buscar artista](https://www.last.fm/music)")
        lfm_oyentes   = st.number_input("Oyentes únicos",    min_value=0, value=0, step=100)
        lfm_scrobbles = st.number_input("Scrobbles totales", min_value=0, value=0, step=1000)

    st.divider()
    col_yt, col_sl = st.columns(2)

    with col_yt:
        st.markdown("**▶️ YouTube**")
        yt_suscriptores   = st.number_input("Suscriptores",   min_value=0, value=0, step=100)
        yt_vistas_totales = st.number_input("Vistas totales", min_value=0, value=0, step=1000)
        yt_num_videos     = st.number_input(
            "Nº vídeos publicados",
            min_value=0, value=0, step=1,
            help="Permite calcular las vistas medias por vídeo. Déjalo en 0 si no lo sabes.",
        )

    with col_sl:
        st.markdown("**🎪 Conciertos**")
        sl_tiene_datos = st.checkbox(
            "El artista aparece en setlist.fm",
            value=False,
            help="Marca si el artista tiene perfil en setlist.fm, aunque no se hayan encontrado conciertos. "
                 "Se activa automáticamente si introduces un número de conciertos > 0.",
        )
        sl_num_conciertos = st.number_input(
            "Nº conciertos documentados en setlist.fm",
            min_value=0, value=0, step=1,
            help="Conciertos registrados. Si el artista no aparece en setlist.fm, deja 0 y desactiva el checkbox.",
        )
        info_conciertos = st.text_area(
            "Info adicional sobre directos (opcional)",
            placeholder="Ej: Tocó en Sala Copera (Granada) con ~150 personas. Ha actuado en festivales locales.",
            help="No entra en el modelo — el agente IA la usa para contextualizar la explicación.",
            height=100,
        )

    submitted = st.form_submit_button("🔍 Predecir tier de sala", width='stretch')

# ---------------------------------------------------------------------------
# Predicción con traza detallada
# ---------------------------------------------------------------------------
if submitted:
    nombre_clean = nombre.strip() if nombre else "Artista"
    log(f"Nueva predicción solicitada para: {nombre_clean}", "STEP")

    with st.status("🔍 Analizando artista...", expanded=True) as status:

        # 1. Inputs recibidos
        st.write("**1 · Inputs recibidos del usuario**")
        inputs_raw = {
            "sp_num_albums":    int(sp_num_albums),
            "sp_num_singles":   int(sp_num_singles),
            "sp_anos_activo":   float(sp_anos_activo),
            "lfm_oyentes":      int(lfm_oyentes),
            "lfm_scrobbles":    int(lfm_scrobbles),
            "yt_suscriptores":  int(yt_suscriptores),
            "yt_vistas_totales":int(yt_vistas_totales),
            "yt_num_videos":    int(yt_num_videos),
            "sl_num_conciertos":int(sl_num_conciertos),
            "sl_tiene_datos":   1 if int(sl_num_conciertos) > 0 else int(sl_tiene_datos),
        }
        for k, v in inputs_raw.items():
            label = FEATURE_LABELS.get(k, k)
            st.write(f"  **{label}**: {v:,}")
        log(f"Inputs: {inputs_raw}", "DATA")

        # 2. Construir las 32 features
        st.write("**2 · Construyendo vector de 32 variables**")
        st.write("  Calculando ratios, logaritmos y rellenando variables no pedidas con valores neutros...")
        features = construir_features(**inputs_raw)
        log("Vector de features construido correctamente", "OK")
        st.write(f"  Lanzamientos/año calculado: **{features['sp_releases_por_ano']:.1f}**")
        st.write(f"  Fidelidad de fans (scrobbles/oyente): **{features['lfm_scrobbles_por_oyente']:.1f}**")
        st.write(f"  Vistas por suscriptor (YouTube): **{features['yt_vistas_por_suscriptor']:.1f}**")
        st.write(f"  Vistas medias por vídeo (YouTube): **{features['yt_vistas_por_video']:.0f}**")
        st.write(f"  Tendencias imputadas a 0 (no se piden al usuario).")

        # 3. Cargar modelo
        st.write("**3 · Cargando modelo**")
        model, meta = _cargar_modelo()
        st.write(f"  Entrenado con **{meta['n_samples']} artistas** y **{meta['n_features']} variables**")
        st.write(f"  Fecha de entrenamiento: {meta['trained_at'][:10]}")
        st.write(f"  Precisión (Accuracy CV5): {meta['cv5_metrics']['acc_mean']:.1%} ± {meta['cv5_metrics']['acc_std']:.1%}")
        st.write(f"  F1 macro CV5: {meta['cv5_metrics']['f1_mean']:.1%} ± {meta['cv5_metrics']['f1_std']:.1%}")
        log(f"Modelo cargado: {meta['n_samples']} artistas, {meta['n_features']} features", "ML")

        # 4. Ejecutar predicción
        st.write("**4 · Ejecutando la predicción**")
        resultado = predecir(**inputs_raw)
        nivel = resultado["nivel"]
        proba = resultado["probabilidades"]
        st.write(f"  Probabilidad BAJO  : {proba['bajo']:.1%}")
        st.write(f"  Probabilidad MEDIO : {proba['medio']:.1%}")
        st.write(f"  Probabilidad ALTO  : {proba['alto']:.1%}")
        st.write(f"  → **Tier predicho: {nivel.upper()}**")
        log(f"Predicción: {nivel.upper()} | bajo={proba['bajo']:.3f} medio={proba['medio']:.3f} alto={proba['alto']:.3f}", "OK")

        status.update(label=f"✅ Predicción completada → {nivel.upper()}", state="complete")

    # Guardar en session_state para el agente IA
    st.session_state["prediccion"] = {
        "resultado":       resultado,
        "nombre":          nombre_clean,
        "info_conciertos": info_conciertos.strip(),
    }
    log(f"Predicción guardada en session_state para {nombre_clean}", "OK")

    # -------------------------------------------------------------------------
    # Resultado visual
    # -------------------------------------------------------------------------
    color = {"bajo": "🔴", "medio": "🟡", "alto": "🟢"}[nivel]
    desc  = {
        "bajo":  "Sala pequeña — menos de 200 personas",
        "medio": "Sala mediana — entre 200 y 2.000 personas",
        "alto":  "Sala grande — más de 2.000 personas / festivales",
    }[nivel]

    st.divider()
    st.subheader(f"{color} Tier predicho: **{nivel.upper()}**")
    st.caption(desc)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Bajo",  f"{proba['bajo']:.1%}")
    c2.metric("🟡 Medio", f"{proba['medio']:.1%}")
    c3.metric("🟢 Alto",  f"{proba['alto']:.1%}")

    st.progress(proba["bajo"],  text=f"Bajo   {proba['bajo']:.1%}")
    st.progress(proba["medio"], text=f"Medio  {proba['medio']:.1%}")
    st.progress(proba["alto"],  text=f"Alto   {proba['alto']:.1%}")

    # -------------------------------------------------------------------------
    # Alertas e inconsistencias
    # -------------------------------------------------------------------------
    alertas = detectar_alertas(features, nivel, proba)
    if alertas:
        st.divider()
        st.markdown("#### ⚡ Avisos del modelo")
        for alerta in alertas:
            if alerta["tipo"] == "warning":
                st.warning(alerta["msg"])
            else:
                st.info(alerta["msg"])

    # -------------------------------------------------------------------------
    # Detalle de variables por fuente — nombres amigables
    # -------------------------------------------------------------------------
    st.divider()
    with st.expander("📊 Ver todas las variables enviadas al modelo (32 variables)"):
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
                (rows_tech if es_tecnica(key) else rows_main).append(
                    {"Variable": label, "Valor": valor}
                )
            if rows_main:
                import pandas as pd
                df_main = pd.DataFrame(rows_main)
                st.dataframe(df_main, hide_index=True, use_container_width=True)
            if rows_tech:
                with st.expander(f"Variables internas del modelo ({len(rows_tech)})", expanded=False):
                    df_tech = pd.DataFrame(rows_tech)
                    st.dataframe(df_tech, hide_index=True, use_container_width=True)
            st.markdown("")

    # -------------------------------------------------------------------------
    # SHAP Waterfall
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 ¿Por qué esta predicción? — Explicación SHAP")
    st.caption(
        "Cada barra muestra cuánto empuja cada variable la predicción hacia el tier predicho. "
        "**Rojo** = empuja hacia arriba (favorece ese tier) · "
        "**Azul** = empuja hacia abajo · "
        "E[f(X)] = predicción media del modelo sobre todos los artistas del dataset."
    )
    try:
        with st.spinner("Calculando valores SHAP..."):
            shap_fig, _ = shap_waterfall_fig(features)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)
        log("Waterfall SHAP generado correctamente", "ML")
    except FileNotFoundError:
        st.info(
            "El modelo serializado no está disponible. "
            "Ejecuta `python -m src.models.train` para entrenarlo.",
            icon="ℹ️",
        )
        log("Modelo no encontrado — SHAP no disponible", "WARN")

    st.divider()
    st.success("✅ Predicción guardada. Ve a **🤖 Analisis IA** para que el agente te explique el resultado.")

    if st.button("🔄 Nueva predicción (limpiar y empezar de nuevo)", width='stretch'):
        reset_prediccion()
        st.rerun()
