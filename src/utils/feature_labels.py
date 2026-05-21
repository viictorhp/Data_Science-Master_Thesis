"""
feature_labels.py
=================
Nombres amigables, agrupación y formateo de las 32 features del modelo.
Detección de inconsistencias y outliers para mostrar alertas en el dashboard.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Nombres legibles por humanos para cada feature técnica
# ---------------------------------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    # Spotify — discografía
    "sp_num_albums":            "Álbumes publicados",
    "sp_num_singles":           "Singles publicados",
    "sp_anos_activo":           "Años en activo",
    "sp_releases_por_ano":      "Lanzamientos por año (calculado)",
    "sp_ratio_albums_singles":  "Ratio álbumes / singles (calculado)",
    "sp_avg_duration_ms":       "Duración media de las canciones",
    "sp_pct_explicit":          "Contenido explícito",
    "sp_pct_colabs":            "Canciones en feat / colaboración",
    # Last.fm
    "lfm_oyentes":              "Oyentes únicos históricos",
    "lfm_oyentes_log":          "Oyentes únicos (escala log, técnico)",
    "lfm_scrobbles":            "Reproducciones acumuladas totales",
    "lfm_scrobbles_log":        "Reproducciones acumuladas (escala log, técnico)",
    "lfm_scrobbles_por_oyente": "Reproducciones por oyente · fidelidad del fan (calculado)",
    "lfm_num_generos":          "Géneros musicales asociados",
    # YouTube
    "yt_suscriptores":          "Suscriptores del canal",
    "yt_suscriptores_log":      "Suscriptores (escala log, técnico)",
    "yt_vistas_totales":        "Vistas totales acumuladas",
    "yt_vistas_log":            "Vistas totales (escala log, técnico)",
    "yt_num_videos":            "Vídeos publicados en el canal",
    "yt_vistas_por_video":      "Vistas medias por vídeo (calculado)",
    "yt_vistas_por_suscriptor": "Vistas por suscriptor · engagement (calculado)",
    # setlist.fm — conciertos
    "sl_num_conciertos":        "Conciertos documentados en setlist.fm",
    "sl_avg_canciones":         "Canciones medias por concierto",
    "sl_pct_encore":            "Conciertos con encore",
    "sl_num_paises":            "Países donde ha actuado",
    "sl_pct_espana":            "Conciertos en España",
    "sl_tiene_datos":           "Aparece en setlist.fm",
    # Tendencias
    "trend_gtrends_interes_medio":        "Interés de búsqueda en Google (últimos 12 meses)",
    "trend_gtrends_log":                  "Interés en Google (escala log, técnico)",
    "trend_yt_vistas_recientes":          "Vistas en los últimos 5 vídeos publicados",
    "trend_yt_vistas_recientes_log":      "Vistas recientes (escala log, técnico)",
    "trend_yt_vistas_por_video_reciente": "Vistas medias por vídeo reciente (calculado)",
}

# ---------------------------------------------------------------------------
# Grupos para mostrar en el dashboard (orden y secciones)
# ---------------------------------------------------------------------------
GRUPOS_DISPLAY: list[tuple[str, list[str]]] = [
    ("🎧 Spotify", [
        "sp_num_albums", "sp_num_singles", "sp_anos_activo",
        "sp_avg_duration_ms", "sp_pct_explicit", "sp_pct_colabs",
        "sp_releases_por_ano", "sp_ratio_albums_singles",
    ]),
    ("📻 Last.fm", [
        "lfm_oyentes", "lfm_scrobbles", "lfm_scrobbles_por_oyente",
        "lfm_num_generos", "lfm_oyentes_log", "lfm_scrobbles_log",
    ]),
    ("▶️ YouTube", [
        "yt_suscriptores", "yt_vistas_totales", "yt_num_videos",
        "yt_vistas_por_video", "yt_vistas_por_suscriptor",
        "yt_suscriptores_log", "yt_vistas_log",
    ]),
    ("🎪 Conciertos (setlist.fm)", [
        "sl_tiene_datos", "sl_num_conciertos", "sl_avg_canciones",
        "sl_pct_encore", "sl_num_paises", "sl_pct_espana",
    ]),
    ("📈 Tendencias actuales", [
        "trend_gtrends_interes_medio", "trend_yt_vistas_recientes",
        "trend_yt_vistas_por_video_reciente",
        "trend_gtrends_log", "trend_yt_vistas_recientes_log",
    ]),
]

# Features que son transformaciones internas del modelo (log, ratios derivados)
_FEATURES_TECNICAS = {
    "sp_releases_por_ano", "sp_ratio_albums_singles",
    "lfm_oyentes_log", "lfm_scrobbles_log", "lfm_scrobbles_por_oyente",
    "yt_suscriptores_log", "yt_vistas_log",
    "yt_vistas_por_video", "yt_vistas_por_suscriptor",
    "trend_gtrends_log", "trend_yt_vistas_recientes_log",
    "trend_yt_vistas_por_video_reciente",
}


def es_tecnica(key: str) -> bool:
    return key in _FEATURES_TECNICAS


def format_valor(key: str, value: float) -> str:
    """Formatea el valor de una feature para mostrarlo de forma legible."""
    if key == "sp_avg_duration_ms":
        total_s = int(value / 1000)
        return f"{total_s // 60}:{total_s % 60:02d} min"
    if key in ("sp_pct_explicit", "sp_pct_colabs"):
        return f"{value:.0%}"
    if key in ("sl_pct_encore", "sl_pct_espana"):
        return f"{value:.0%}"
    if key == "sl_tiene_datos":
        return "✅ Sí" if value >= 0.5 else "❌ No"
    if key.endswith("_log"):
        return f"{value:.3f}  *(interno)*"
    if key in ("sp_releases_por_ano", "sp_ratio_albums_singles",
               "lfm_scrobbles_por_oyente", "yt_vistas_por_video",
               "yt_vistas_por_suscriptor", "trend_yt_vistas_por_video_reciente"):
        return f"{value:.2f}  *(calculado)*"
    if key in ("lfm_oyentes", "lfm_scrobbles", "yt_suscriptores", "yt_vistas_totales",
               "sl_num_conciertos", "trend_yt_vistas_recientes"):
        return f"{int(value):,}"
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"


def build_grupo_df(keys: list[str], features: dict) -> pd.DataFrame:
    """Construye el DataFrame de una sección para st.dataframe."""
    rows = []
    for key in keys:
        if key not in features:
            continue
        label = FEATURE_LABELS.get(key, key)
        valor = format_valor(key, features[key])
        tecnica = "Sí" if es_tecnica(key) else ""
        rows.append({"Métrica": label, "Valor": valor, "Interna": tecnica})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detección de alertas e inconsistencias
# ---------------------------------------------------------------------------

def detectar_alertas(features: dict, nivel: str, proba: dict) -> list[dict]:
    """
    Analiza la predicción y devuelve una lista de alertas con tipo y mensaje.
    tipo: "warning" | "info"
    """
    alertas = []

    lfm_o  = features.get("lfm_oyentes", 0)
    yt_s   = features.get("yt_suscriptores", 0)
    yt_v   = features.get("yt_vistas_totales", 0)
    yt_nv  = features.get("yt_num_videos", 0)
    sl_c   = features.get("sl_num_conciertos", 0)
    sl_td  = features.get("sl_tiene_datos", 0)
    rpm    = features.get("sp_releases_por_ano", 0)
    conf   = proba[nivel]

    # ── Confianza ─────────────────────────────────────────────────────────────
    adyacentes = {"bajo": ["medio"], "medio": ["bajo", "alto"], "alto": ["medio"]}
    candidatos = {k: v for k, v in proba.items() if k in adyacentes[nivel]}
    segundo = max(candidatos.items(), key=lambda x: x[1])
    margen = conf - segundo[1]

    if conf < 0.55:
        alertas.append({
            "tipo": "warning",
            "msg": (
                f"**Predicción con baja confianza ({conf:.1%}).**  \n"
                f"El modelo duda entre **{nivel.upper()}** ({conf:.1%}) "
                f"y **{segundo[0].upper()}** ({segundo[1]:.1%}). "
                "Este es un artista en zona frontera — pequeños cambios en las métricas "
                "podrían invertir el resultado."
            ),
        })
    elif conf < 0.70:
        alertas.append({
            "tipo": "info",
            "msg": (
                f"**Predicción con confianza media ({conf:.1%}).**  \n"
                f"El modelo se inclina por **{nivel.upper()}** ({conf:.1%}), "
                f"pero **{segundo[0].upper()}** tiene una probabilidad relevante ({segundo[1]:.1%}). "
                "El resultado es orientativo — complementa con el análisis SHAP."
            ),
        })
    elif margen < 0.15:
        alertas.append({
            "tipo": "info",
            "msg": (
                f"**Margen estrecho entre los dos tiers más probables.**  \n"
                f"Aunque la confianza es alta ({conf:.1%}), **{segundo[0].upper()}** "
                f"está a solo {margen:.1%} de diferencia. "
                "Un par de métricas adicionales podría mover la predicción."
            ),
        })

    # ── Presencia digital muy superior al tier predicho ───────────────────────
    if nivel == "bajo":
        razones = []
        if lfm_o > 150_000:
            razones.append(f"**{lfm_o:,.0f} oyentes** en Last.fm → rango típico de **ALTO** (>150K)")
        elif lfm_o > 50_000:
            razones.append(f"**{lfm_o:,.0f} oyentes** en Last.fm → rango típico de **MEDIO** (15K–150K)")
        if yt_s > 200_000:
            razones.append(f"**{yt_s:,.0f} suscriptores** en YouTube → rango típico de **ALTO** (>200K)")
        elif yt_s > 30_000:
            razones.append(f"**{yt_s:,.0f} suscriptores** en YouTube → rango típico de **MEDIO** (10K–200K)")
        if yt_v > 10_000_000:
            razones.append(f"**{yt_v:,.0f} vistas** en YouTube → rango típico de **ALTO** (>10M)")
        elif yt_v > 500_000:
            razones.append(f"**{yt_v:,.0f} vistas** en YouTube → rango típico de **MEDIO** (>500K)")
        if razones:
            alertas.append({
                "tipo": "warning",
                "msg": (
                    "**La presencia digital supera lo esperado para el tier BAJO predicho.**  \n"
                    "La ausencia de datos de conciertos está arrastrando la predicción hacia abajo, "
                    "pero estas métricas apuntan a un tier mayor:  \n"
                    + "  \n".join(f"• {r}" for r in razones)
                ),
            })

    if nivel in ("bajo", "medio"):
        razones_alto = []
        if lfm_o > 300_000:
            razones_alto.append(f"**{lfm_o:,.0f} oyentes** en Last.fm (rango ALTO: >150K)")
        if yt_s > 500_000:
            razones_alto.append(f"**{yt_s:,.0f} suscriptores** en YouTube (rango ALTO: >200K)")
        if yt_v > 100_000_000:
            razones_alto.append(f"**{yt_v:,.0f} vistas** en YouTube (rango ALTO: >10M)")
        if razones_alto:
            alertas.append({
                "tipo": "warning",
                "msg": (
                    f"**Métricas en rango ALTO pese a la predicción {nivel.upper()}:**  \n"
                    + "  \n".join(f"• {r}" for r in razones_alto)
                ),
            })

    # ── Sin conciertos pero alta presencia digital ────────────────────────────
    if sl_c == 0 and sl_td == 0 and (lfm_o > 20_000 or yt_s > 15_000):
        alertas.append({
            "tipo": "info",
            "msg": (
                "**El artista no aparece en setlist.fm y no se han registrado conciertos.**  \n"
                "Los conciertos documentados son la variable más determinante del modelo. "
                "Si el artista ha actuado en directo, introducir esos datos podría cambiar "
                "significativamente la predicción."
            ),
        })

    # ── yt_num_videos = 0 con vistas altas → yt_vistas_por_video = 0 ─────────
    if yt_nv == 0 and yt_v > 500_000:
        alertas.append({
            "tipo": "info",
            "msg": (
                f"**No se introdujo el número de vídeos del canal de YouTube**, "
                f"por lo que la métrica 'vistas medias por vídeo' es 0 aunque el canal "
                f"acumula {yt_v:,.0f} vistas totales. Esto puede subestimar la relevancia del canal."
            ),
        })

    # ── Ritmo de lanzamientos anómalo ────────────────────────────────────────
    if rpm > 20:
        alertas.append({
            "tipo": "info",
            "msg": (
                f"**Ritmo de lanzamientos muy alto ({rpm:.0f} lanzamientos/año).**  \n"
                "Esto puede indicar que los 'años en activo' están subestimados, "
                "o que los datos de Spotify incluyen recopilaciones o colaboraciones."
            ),
        })

    return alertas
