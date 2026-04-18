"""
Recoge popularidad y seguidores de artistas directamente del endpoint artist de Spotify.

Genera: data/raw/spotify_artistas.csv
  - popularidad:  score 0-100 de Spotify (señal muy predictiva)
  - seguidores:   followers.total del artista

Uso:
  python -m src.data_collectors.spotify_artistas
  python -m src.data_collectors.spotify_artistas --force Dano "Rels B"
"""

import os
import time
import argparse
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DISCOGRAFIA_CSV = Path("data/raw/spotify_discografia.csv")
OUTPUT_CSV      = Path("data/raw/spotify_artistas.csv")
DELAY           = 0.3


def _init_sp() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))


def _safe_artist(sp: spotipy.Spotify, artist_id: str, retries: int = 3) -> dict | None:
    for i in range(retries):
        try:
            return sp.artist(artist_id)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait = int(e.headers.get("Retry-After", 10))
                print(f"  [rate limit] esperando {wait}s...")
                time.sleep(wait)
            elif e.http_status in (400, 403, 404):
                return None
            else:
                print(f"  [error {e.http_status}] reintentando...")
                time.sleep(5)
        except Exception as e:
            print(f"  [error red] reintentando... ({e})")
            time.sleep(5 * (i + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Recoge popularidad y seguidores de Spotify")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    if not DISCOGRAFIA_CSV.exists():
        print(f"No se encuentra {DISCOGRAFIA_CSV}. Ejecuta spotify_features.py primero.")
        return

    df_ids = pd.read_csv(DISCOGRAFIA_CSV)[["nombre_buscado", "artist_id"]].dropna(subset=["artist_id"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df_existe = pd.read_csv(OUTPUT_CSV) if OUTPUT_CSV.exists() else pd.DataFrame()

    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        borrados = df_existe[df_existe["nombre_buscado"].str.lower().isin(force_lower)]["nombre_buscado"].tolist()
        df_existe = df_existe[~df_existe["nombre_buscado"].str.lower().isin(force_lower)]
        if borrados:
            df_existe.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            print(f"Re-procesando: {borrados}")

    procesados = set(df_existe["nombre_buscado"].str.lower()) if not df_existe.empty else set()
    pendientes = df_ids[~df_ids["nombre_buscado"].str.lower().isin(procesados)]

    print(f"Ya procesados: {len(procesados)} | Pendientes: {len(pendientes)}")
    if pendientes.empty:
        print("Todos los artistas ya están procesados.")
        return

    sp = _init_sp()
    print("✓ Cliente Spotify inicializado\n")

    filas = []
    total = len(pendientes)

    for i, (_, row) in enumerate(pendientes.iterrows(), 1):
        nombre    = row["nombre_buscado"]
        artist_id = row["artist_id"]
        print(f"[{i}/{total}] {nombre} ...", end=" ", flush=True)

        data = _safe_artist(sp, artist_id)
        if not data:
            print("error/no encontrado")
            filas.append({"nombre_buscado": nombre, "popularidad": None, "seguidores": None})
        else:
            pop = data.get("popularity")
            seg = data.get("followers", {}).get("total")
            seg_fmt = f"{seg:,}" if seg is not None else "N/A"
            print(f"popularidad={pop} | seguidores={seg_fmt}")
            filas.append({"nombre_buscado": nombre, "popularidad": pop, "seguidores": seg})

        time.sleep(DELAY)

    df_nuevo  = pd.DataFrame(filas)
    df_final  = pd.concat([df_existe, df_nuevo], ignore_index=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓ {len(filas)} artistas guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
