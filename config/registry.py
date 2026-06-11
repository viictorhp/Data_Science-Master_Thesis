"""
Fuente de verdad para identidades de artistas entre plataformas.

Columnas del registry:
  nombre_canonico  — nombre canónico del artista (clave primaria)
  spotify_id       — Spotify artist ID
  spotify_nombre   — nombre tal como lo devuelve Spotify
  spotify_score    — puntuación del match (0.0 al 1.0)
  lastfm_nombre    — nombre tal como lo usa Last.fm
  lastfm_mbid      — MusicBrainz ID desde Last.fm
  lastfm_score     — puntuación del match
  yt_channel_id    — YouTube channel ID (UC...)
  yt_nombre_canal  — nombre del canal
  yt_score         — puntuación del match (1.0 = corrección manual)
  sl_nombre        — nombre tal como lo usa setlist.fm
  sl_mbid          — MBID desde setlist.fm
  sl_score         — puntuación del match
  manual_override  — True = no sobreescribir automáticamente esta fila
  notas            — motivo de corrección manual u observaciones
"""

from pathlib import Path
import pandas as pd

REGISTRY_PATH = Path(__file__).parent / "artistas_registry.csv"

COLUMNS = [
    "nombre_canonico",
    "spotify_id",    "spotify_nombre",  "spotify_score",
    "lastfm_nombre", "lastfm_mbid",     "lastfm_score",
    "yt_channel_id", "yt_nombre_canal", "yt_score",
    "sl_nombre",     "sl_mbid",         "sl_score",
    "manual_override", "notas",
]


def cargar() -> pd.DataFrame:
    """Carga el registry. Lanza FileNotFoundError si no existe."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"{REGISTRY_PATH} no encontrado.\n"
            "  Primera vez: python scripts/migrar_registry.py\n"
            "  Artistas nuevos: python scripts/00_resolver_identidades.py"
        )
    df = pd.read_csv(REGISTRY_PATH, dtype=str)
    df["manual_override"] = df["manual_override"].fillna("False").map(
        {"True": True, "False": False, "true": True, "false": False}
    ).fillna(False)
    return df


def guardar(df: pd.DataFrame) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).to_csv(REGISTRY_PATH, index=False, encoding="utf-8")
