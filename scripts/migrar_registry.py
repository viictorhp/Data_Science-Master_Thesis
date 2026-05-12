"""
Migración inicial: construye config/artistas_registry.csv desde los CSVs ya existentes.

Ejecutar UNA SOLA VEZ tras instalar el nuevo sistema de registry.
No consume cuota de API — lee los ficheros que ya tienes en data/raw/.

Regla de seguridad:
  Todos los artistas que ya tienen algún dato en los CSVs se marcan con
  manual_override=True. Esto garantiza que scripts/00_resolver_identidades.py
  nunca sobreescriba datos que ya tenías (incluidas tus correcciones manuales).
  Solo quedan con manual_override=False los artistas sin ningún dato todavía.

Uso:
  python scripts/migrar_registry.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.registry import COLUMNS, guardar

RAW = ROOT / "data" / "raw"


def _leer_csv_seguro(path: Path, cols: list[str]) -> pd.DataFrame | None:
    """Lee un CSV y selecciona solo las columnas que realmente existen."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols_presentes = [c for c in cols if c in df.columns]
    if not cols_presentes:
        return None
    return df[cols_presentes]


def main():
    artistas_file = ROOT / "artistas.txt"
    with open(artistas_file, encoding="utf-8") as f:
        artistas = [l.strip() for l in f if l.strip()]
    print(f"{len(artistas)} artistas en artistas.txt")

    df = pd.DataFrame({"nombre_canonico": artistas, "manual_override": False, "notas": ""})

    # ── Spotify ───────────────────────────────────────────────────────────────
    df_sp = _leer_csv_seguro(RAW / "spotify_ids.csv",
                              ["nombre_buscado", "artist_id", "nombre_spotify", "match_score"])
    if df_sp is not None:
        df_sp = df_sp.rename(columns={
            "nombre_buscado": "nombre_canonico",
            "artist_id":      "spotify_id",
            "nombre_spotify": "spotify_nombre",
            "match_score":    "spotify_score",
        })
        df = df.merge(df_sp, on="nombre_canonico", how="left")
        print(f"  Spotify:    {df['spotify_id'].notna().sum()}/{len(df)} artistas")
    else:
        print("  Spotify:    spotify_ids.csv no encontrado — omitido")

    # ── Last.fm ───────────────────────────────────────────────────────────────
    df_lf = _leer_csv_seguro(RAW / "lastfm_artistas.csv",
                              ["nombre_buscado", "nombre_lastfm", "mbid", "match_score"])
    if df_lf is not None:
        df_lf = df_lf.rename(columns={
            "nombre_buscado": "nombre_canonico",
            "nombre_lastfm":  "lastfm_nombre",
            "mbid":           "lastfm_mbid",
            "match_score":    "lastfm_score",
        })
        df = df.merge(df_lf, on="nombre_canonico", how="left")
        print(f"  Last.fm:    {df['lastfm_nombre'].notna().sum()}/{len(df)} artistas")
    else:
        print("  Last.fm:    lastfm_artistas.csv no encontrado — omitido")

    # ── YouTube ───────────────────────────────────────────────────────────────
    df_yt = _leer_csv_seguro(RAW / "youtube_artistas.csv",
                              ["nombre_buscado", "channel_id", "nombre_canal", "match_score"])
    if df_yt is not None:
        df_yt = df_yt.rename(columns={
            "nombre_buscado": "nombre_canonico",
            "channel_id":     "yt_channel_id",
            "nombre_canal":   "yt_nombre_canal",
            "match_score":    "yt_score",
        })
        df = df.merge(df_yt, on="nombre_canonico", how="left")
        print(f"  YouTube:    {df['yt_channel_id'].notna().sum()}/{len(df)} artistas")
    else:
        print("  YouTube:    youtube_artistas.csv no encontrado — omitido")

    # ── setlist.fm ────────────────────────────────────────────────────────────
    df_sl_raw = _leer_csv_seguro(RAW / "setlistfm_conciertos.csv", ["nombre", "setlistfm_mbid"])
    if df_sl_raw is not None:
        df_sl = (
            df_sl_raw[df_sl_raw["setlistfm_mbid"].notna()]
            .drop_duplicates(subset=["nombre"])
            .rename(columns={"nombre": "nombre_canonico", "setlistfm_mbid": "sl_mbid"})
        )
        df_sl["sl_nombre"] = df_sl["nombre_canonico"]
        df_sl["sl_score"]  = 1.0
        df = df.merge(df_sl, on="nombre_canonico", how="left")
        print(f"  setlist.fm: {df['sl_mbid'].notna().sum()}/{len(df)} artistas")
    else:
        print("  setlist.fm: setlistfm_conciertos.csv no encontrado — omitido")

    # ── Proteger datos existentes ─────────────────────────────────────────────
    # Cualquier artista que ya tiene al menos un ID importado se marca como
    # manual_override=True: sus datos no serán sobreescritos por el resolver.
    campos_id = ["spotify_id", "lastfm_nombre", "yt_channel_id", "sl_mbid"]
    # Añadir columnas que no se crearon (ningún CSV encontrado para esa plataforma)
    for col in campos_id:
        if col not in df.columns:
            df[col] = None
    tiene_datos = df[campos_id].notna().any(axis=1)
    df.loc[tiene_datos, "manual_override"] = True

    n_protegidos = tiene_datos.sum()
    n_sin_datos  = (~tiene_datos).sum()
    print(f"\n  {n_protegidos} artistas con datos existentes → manual_override=True (protegidos)")
    print(f"  {n_sin_datos} artistas sin ningún dato → manual_override=False (pendientes de resolver)")

    # ── Guardar ───────────────────────────────────────────────────────────────
    guardar(df)
    print(f"\nRegistry guardado: config/artistas_registry.csv ({len(df)} artistas)")

    sin_datos_df = df[~tiene_datos]
    if not sin_datos_df.empty:
        print(f"\nArtistas pendientes de resolver ({n_sin_datos}):")
        for nombre in sin_datos_df["nombre_canonico"].tolist():
            print(f"  - {nombre}")
        print("\nEjecuta: python scripts/00_resolver_identidades.py")


if __name__ == "__main__":
    main()
