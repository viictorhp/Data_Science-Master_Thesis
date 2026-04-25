"""
build_features.py
=================
Construye la matriz de features por artista combinando todas las fuentes raw.

Fuentes:
  - artistas_labels.csv      → target (nivel: bajo/medio/alto, tier_sala: 1/2/3)
  - spotify_discografia.csv  → discografía, cadencia, actividad reciente,
                                ratio madurez de carrera
  - spotify_tracks.csv       → popularidad de tracks, duración, explicit, collabs
  - lastfm_artistas.csv      → oyentes, scrobbles, engagement, géneros
  - youtube_artistas.csv     → audiencia digital, engagement, antigüedad canal
  - setlistfm_conciertos.csv → actividad en directo, geografía, detalle de set
  - tendencias.csv           → interés de búsqueda (Google Trends) y actividad
                                reciente en YouTube (últimos 5 vídeos)

Salida:
  data/processed/artist_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
HOY       = pd.Timestamp("today").normalize()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load(filename: str) -> pd.DataFrame:
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichero no encontrado: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Agregadores por fuente
# ---------------------------------------------------------------------------

def _features_spotify_discografia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volumen de releases, ritmo de producción, madurez de carrera y actividad reciente.
    sp_ratio_albums_singles: alto = carrera consolidada; bajo = estrategia streaming-first.
    sp_dias_desde_ultimo_release: artista activo vs estancado.
    """
    df = df.copy()
    df["ultimo_lanzamiento"] = pd.to_datetime(df["ultimo_lanzamiento"], errors="coerce")
    df["sp_dias_desde_ultimo_release"] = (HOY - df["ultimo_lanzamiento"]).dt.days
    df["sp_ratio_albums_singles"]      = df["num_albums"] / (df["num_singles"] + 1)

    return df[[
        "nombre_buscado",
        "num_albums", "num_singles", "num_eps", "num_total_releases",
        "anos_activo", "releases_por_ano",
        "sp_dias_desde_ultimo_release", "sp_ratio_albums_singles",
    ]].rename(columns={
        "num_albums":         "sp_num_albums",
        "num_singles":        "sp_num_singles",
        "num_eps":            "sp_num_eps",
        "num_total_releases": "sp_num_total_releases",
        "anos_activo":        "sp_anos_activo",
        "releases_por_ano":   "sp_releases_por_ano",
    })


def _features_spotify_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duración media de tracks, % explicit y % colaboraciones.
    (popularity bloqueada por Spotify desde nov 2024 — eliminada)
    """
    df = df.copy()
    df["explicit"] = df["explicit"].map({"True": True, "False": False, True: True, False: False})
    df["es_colab"] = df["num_artistas"] > 1

    agg = df.groupby("artist_id").agg(
        sp_avg_duration_ms =("duration_ms", "mean"),
        sp_pct_explicit    =("explicit",    "mean"),
        sp_pct_colabs      =("es_colab",    "mean"),
    ).reset_index()

    name_map = df[["nombre_artista", "artist_id"]].drop_duplicates()
    agg = agg.merge(name_map, on="artist_id", how="left")
    return agg.rename(columns={"nombre_artista": "nombre_buscado"}).drop(columns=["artist_id"])


def _features_lastfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audiencia Last.fm: oyentes mensuales, volumen histórico de scrobbles,
    engagement (scrobbles por oyente) y riqueza de género.
    Versiones log para compensar la fuerte asimetría de las distribuciones.
    """
    df = df.copy()
    df["oyentes"]   = pd.to_numeric(df["oyentes"],   errors="coerce")
    df["scrobbles"] = pd.to_numeric(df["scrobbles"], errors="coerce")

    df["lfm_oyentes"]              = df["oyentes"]
    df["lfm_scrobbles"]            = df["scrobbles"]
    df["lfm_oyentes_log"]          = np.log1p(df["oyentes"])
    df["lfm_scrobbles_log"]        = np.log1p(df["scrobbles"])
    df["lfm_scrobbles_por_oyente"] = df["scrobbles"] / df["oyentes"].replace(0, np.nan)
    df["lfm_num_generos"] = (
        df["generos"]
        .fillna("")
        .apply(lambda g: 0 if g in ("", "Sin género") else len(g.split(",")))
    )

    return df[[
        "nombre_buscado",
        "lfm_oyentes", "lfm_oyentes_log",
        "lfm_scrobbles", "lfm_scrobbles_log",
        "lfm_scrobbles_por_oyente", "lfm_num_generos",
    ]]


def _features_youtube(df: pd.DataFrame) -> pd.DataFrame:
    """
    Presencia digital en YouTube: tamaño de audiencia, actividad,
    engagement y antigüedad del canal.
    Versiones log para suscriptores y vistas por su alta asimetría.
    yt_vistas_por_video / yt_vistas_por_suscriptor: engagement real del canal.
    """
    df = df.copy()
    for col in ["suscriptores", "vistas_totales", "num_videos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fecha_creacion"]           = pd.to_datetime(df["fecha_creacion"], errors="coerce")
    df["yt_edad_canal_anos"]       = (HOY - df["fecha_creacion"]).dt.days / 365.25
    df["yt_suscriptores"]          = df["suscriptores"]
    df["yt_suscriptores_log"]      = np.log1p(df["suscriptores"])
    df["yt_vistas_totales"]        = df["vistas_totales"]
    df["yt_vistas_log"]            = np.log1p(df["vistas_totales"])
    df["yt_num_videos"]            = df["num_videos"]
    df["yt_vistas_por_video"]      = df["vistas_totales"] / df["num_videos"].replace(0, np.nan)
    df["yt_vistas_por_suscriptor"] = df["vistas_totales"] / df["suscriptores"].replace(0, np.nan)

    return df[[
        "nombre_buscado",
        "yt_suscriptores", "yt_suscriptores_log",
        "yt_vistas_totales", "yt_vistas_log",
        "yt_num_videos",
        "yt_vistas_por_video", "yt_vistas_por_suscriptor",
        "yt_edad_canal_anos",
    ]]


def _features_tendencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Señales de tendencia actuales: interés de búsqueda en Google (España, 12 meses)
    y actividad reciente en YouTube (vistas + vídeos últimos 5 uploads).
    trend_yt_vistas_por_video_reciente: engagement reciente por vídeo, más sensible
    que las vistas totales del canal para detectar artistas en ascenso.
    """
    df = df.copy()
    for col in ["gtrends_interes_medio", "yt_vistas_recientes", "yt_videos_recientes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trend_gtrends_interes_medio"]        = df["gtrends_interes_medio"]
    df["trend_gtrends_log"]                  = np.log1p(df["gtrends_interes_medio"])
    df["trend_yt_vistas_recientes"]          = df["yt_vistas_recientes"]
    df["trend_yt_vistas_recientes_log"]      = np.log1p(df["yt_vistas_recientes"])
    df["trend_yt_videos_recientes"]          = df["yt_videos_recientes"]
    df["trend_yt_vistas_por_video_reciente"] = (
        df["yt_vistas_recientes"] / df["yt_videos_recientes"].replace(0, np.nan)
    )

    return df[[
        "nombre_buscado",
        "trend_gtrends_interes_medio", "trend_gtrends_log",
        "trend_yt_vistas_recientes", "trend_yt_vistas_recientes_log",
        "trend_yt_videos_recientes",
        "trend_yt_vistas_por_video_reciente",
    ]]


def _features_setlistfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Actividad en directo: volumen de conciertos, calidad del set (canciones,
    encore) y geografía (países, % en España como proxy de proyección internacional).
    """
    df = df.copy()
    df["num_canciones"] = pd.to_numeric(df["num_canciones"], errors="coerce")
    df["tiene_encore"]  = df["tiene_encore"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    # Solo filas con fecha confirmada — evita contar filas sin setlist real
    df_valido = df[df["fecha"].notna()].copy()

    if df_valido.empty:
        return pd.DataFrame(columns=[
            "nombre_buscado",
            "sl_num_conciertos", "sl_avg_canciones", "sl_pct_encore",
            "sl_num_paises", "sl_pct_espana",
        ])

    # Usar size() para contar filas reales (count("setlist_id") da 0 si setlist_id es NaN)
    conteo = df_valido.groupby("nombre").size().rename("sl_num_conciertos")
    agg = df_valido.groupby("nombre").agg(
        sl_avg_canciones =("num_canciones", "mean"),
        sl_pct_encore    =("tiene_encore",  "mean"),
        sl_num_paises    =("pais",          "nunique"),
    ).reset_index().rename(columns={"nombre": "nombre_buscado"})
    agg = agg.join(conteo, on="nombre_buscado")

    conc_espana = (
        df_valido[df_valido["pais"] == "Spain"]
        .groupby("nombre")
        .size()
        .rename("_conc_espana")
    )
    agg = agg.join(conc_espana, on="nombre_buscado")
    agg["sl_pct_espana"] = agg["_conc_espana"].fillna(0) / agg["sl_num_conciertos"]
    return agg.drop(columns=["_conc_espana"])


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def build_artist_features() -> pd.DataFrame:
    """Carga todas las fuentes y devuelve la matriz de features con el target."""
    labels = _load("artistas_labels.csv")[["nombre_buscado", "nivel", "tier_sala"]]
    labels["target"] = labels["nivel"].map({"bajo": 1, "medio": 2, "alto": 3})

    sources = [
        _features_spotify_discografia(_load("spotify_discografia.csv")),
        _features_spotify_tracks(_load("spotify_tracks.csv")),
        _features_lastfm(_load("lastfm_artistas.csv")),
        _features_youtube(_load("youtube_artistas.csv")),
        _features_setlistfm(_load("setlistfm_conciertos.csv")),
        _features_tendencias(_load("tendencias.csv")),
    ]

    df = labels.copy()
    for source in sources:
        df = df.merge(source, on="nombre_buscado", how="left")
    return df


def save_features(df: pd.DataFrame) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "artist_features.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_artist_features()

    print(f"Shape: {df.shape}")
    print(f"\nColumnas ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"\nDistribución del target:")
    print(df["nivel"].value_counts())
    print(f"\nNaN por columna (%):")
    nan_pct = (df.isna().sum() / len(df) * 100).round(1)
    print(nan_pct[nan_pct > 0].sort_values(ascending=False))

    out = save_features(df)
    print(f"\nGuardado en: {out}")
