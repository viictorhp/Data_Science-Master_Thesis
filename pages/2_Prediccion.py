import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

import streamlit as st

from src.models.predict import construir_features, predecir, _cargar_modelo
from src.utils.log_streamlit import log, render_sidebar_log, reset_prediccion

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

    with col_sl:
        st.markdown("**🎪 Conciertos**")
        sl_num_conciertos = st.number_input(
            "Nº conciertos documentados en setlist.fm",
            min_value=0, value=0, step=1,
            help="Si el artista no aparece en setlist.fm, deja 0.",
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
            "sl_num_conciertos":int(sl_num_conciertos),
        }
        for k, v in inputs_raw.items():
            st.write(f"  `{k}` = **{v:,}**")
        log(f"Inputs: {inputs_raw}", "DATA")

        # 2. Construir las 31 features
        st.write("**2 · Construyendo vector de 31 features**")
        st.write("  Calculando ratios, logs y rellenando campos no pedidos con valores neutros...")
        features = construir_features(**inputs_raw)
        log("Vector de features construido correctamente", "OK")

        calculadas = {
            "sp_releases_por_ano":      features["sp_releases_por_ano"],
            "sp_ratio_albums_singles":  features["sp_ratio_albums_singles"],
            "lfm_oyentes_log":          features["lfm_oyentes_log"],
            "lfm_scrobbles_log":        features["lfm_scrobbles_log"],
            "lfm_scrobbles_por_oyente": features["lfm_scrobbles_por_oyente"],
            "yt_suscriptores_log":      features["yt_suscriptores_log"],
            "yt_vistas_log":            features["yt_vistas_log"],
            "yt_vistas_por_suscriptor": features["yt_vistas_por_suscriptor"],
        }
        st.write("  Derivadas calculadas:")
        for k, v in calculadas.items():
            st.write(f"    `{k}` = {v:.4f}")
        st.write(f"  Tendencias imputadas a 0 (no se piden al usuario).")

        # 3. Cargar modelo
        st.write("**3 · Cargando modelo**")
        model, meta = _cargar_modelo()
        st.write(f"  Archivo: `models/xgb_tuned.joblib`")
        st.write(f"  Entrenado: {meta['trained_at'][:19]}")
        st.write(f"  Artistas de entrenamiento: {meta['n_samples']}")
        st.write(f"  Features esperadas: {meta['n_features']}")
        st.write(f"  Accuracy CV5: {meta['cv5_metrics']['acc_mean']:.1%} ± {meta['cv5_metrics']['acc_std']:.1%}")
        st.write(f"  F1 macro CV5: {meta['cv5_metrics']['f1_mean']:.1%} ± {meta['cv5_metrics']['f1_std']:.1%}")
        log(f"Modelo cargado: {meta['n_samples']} artistas, {meta['n_features']} features", "ML")

        # 4. Ejecutar predicción
        st.write("**4 · Ejecutando predict_proba**")
        resultado = predecir(**inputs_raw)
        nivel = resultado["nivel"]
        proba = resultado["probabilidades"]
        st.write(f"  Bajo  : {proba['bajo']:.4f}")
        st.write(f"  Medio : {proba['medio']:.4f}")
        st.write(f"  Alto  : {proba['alto']:.4f}")
        st.write(f"  → Clase predicha: **{nivel.upper()}** (argmax)")
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

    # Expander con el vector completo de 31 features
    with st.expander("🔬 Ver vector completo de features enviado al modelo (31 features)"):
        col_a, col_b = st.columns(2)
        items = list(features.items())
        mid   = len(items) // 2
        for k, v in items[:mid]:
            col_a.text(f"{k:<35} {v:.6g}")
        for k, v in items[mid:]:
            col_b.text(f"{k:<35} {v:.6g}")

    st.divider()
    st.success("✅ Predicción guardada. Ve a **🤖 Analisis IA** para que el agente te explique el resultado.")

    if st.button("🔄 Nueva predicción (limpiar y empezar de nuevo)", width='stretch'):
        reset_prediccion()
        st.rerun()
