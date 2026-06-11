"""
Extrae features de Spotify por artista usando los IDs del registry.

Genera dos CSVs:
  - spotify_discografia.csv   → una fila por artista con stats de discografía
  - spotify_tracks.csv        → una fila por canción (top 10 por artista)

Variables generadas:
  Discografía: num_albums, num_singles, num_eps, num_total_releases,
               primer_lanzamiento, ultimo_lanzamiento, anos_activo,
               releases_por_ano
  Top tracks:  track_name, duration_ms, explicit, num_artistas

Uso:
  python -m src.data_collectors.spotify_features
  python -m src.data_collectors.spotify_features --force Dano "Rels B"
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DISCOGRAFIA_CSV = ROOT / "data" / "raw" / "spotify_discografia.csv"
TOP_TRACKS_CSV  = ROOT / "data" / "raw" / "spotify_tracks.csv"
DISCOGRAFIA_CSV.parent.mkdir(parents=True, exist_ok=True)

DELAY      = 0.5
ANO_ACTUAL = datetime.now().year


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_call(sp, fn, *args, retries=3, silent_4xx=True, **kwargs):
    """Llama a la API con reintentos ante errores de red o rate limit."""
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait = int(e.headers.get("Retry-After", 10))
                print(f"  [rate limit] esperando {wait}s...")
                time.sleep(wait)
            elif e.http_status in (400, 403, 404):
                if not silent_4xx:
                    print(f"  [Spotify {e.http_status}] {e}")
                return None
            else:
                print(f"  [Spotify error {e.http_status}] {e}")
                time.sleep(5)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            print(f"  [error red {type(e).__name__}] reintentando...")
            time.sleep(5 * (i + 1))
    return None


def _get_albums_page(sp, artist_id: str, offset: int) -> dict | None:
    """Obtiene una página de álbumes via requests con market y sin encode de comas."""
    token = sp.auth_manager.get_access_token(as_dict=False)
    url = (
        f"https://api.spotify.com/v1/artists/{artist_id}/albums"
        f"?include_groups=album,single&market=ES&limit=10&offset={offset}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  [rate limit] esperando {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code in (400, 403, 404):
                print(f"  [albums error {r.status_code}] {r.text[:200]}")
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  [error red] {e}")
            time.sleep(5 * (attempt + 1))
    return None


def get_discografia(sp, artist_id: str, nombre: str) -> dict:
    """Obtiene stats de discografía via artists/{id}/albums."""
    albums_raw, singles_raw, eps_raw = [], [], []
    offset = 0
    while True:
        res = _get_albums_page(sp, artist_id, offset)
        if not res:
            break
        items = res.get("items", [])
        if not items:
            break
        for item in items:
            album_type   = item.get("album_type", "")
            release_date = item.get("release_date", "")
            if album_type == "album":
                albums_raw.append(release_date)
            elif album_type == "single":
                singles_raw.append(release_date)
            elif album_type in ("compilation", "ep"):
                eps_raw.append(release_date)
        if not res.get("next"):
            break
        offset += 50
        time.sleep(DELAY)

    todas_fechas = [f for f in albums_raw + singles_raw + eps_raw if f]
    todas_fechas_dt = []
    for f in todas_fechas:
        try:
            if len(f) == 4:
                todas_fechas_dt.append(datetime(int(f), 1, 1))
            elif len(f) == 7:
                todas_fechas_dt.append(datetime(int(f[:4]), int(f[5:7]), 1))
            else:
                todas_fechas_dt.append(datetime.strptime(f, "%Y-%m-%d"))
        except Exception:
            pass

    primer_lanzamiento = min(todas_fechas_dt).strftime("%Y-%m-%d") if todas_fechas_dt else None
    ultimo_lanzamiento  = max(todas_fechas_dt).strftime("%Y-%m-%d") if todas_fechas_dt else None
    anos_activo         = None
    releases_por_ano    = None
    if todas_fechas_dt:
        ano_inicio       = min(todas_fechas_dt).year
        anos_activo      = max(1, ANO_ACTUAL - ano_inicio)
        num_total        = len(albums_raw) + len(singles_raw) + len(eps_raw)
        releases_por_ano = round(num_total / anos_activo, 2)

    return {
        "num_albums":         len(albums_raw),
        "num_singles":        len(singles_raw),
        "num_eps":            len(eps_raw),
        "num_total_releases": len(albums_raw) + len(singles_raw) + len(eps_raw),
        "primer_lanzamiento": primer_lanzamiento,
        "ultimo_lanzamiento": ultimo_lanzamiento,
        "anos_activo":        anos_activo,
        "releases_por_ano":   releases_por_ano,
    }


def get_spotify_tracks(sp, artist_id: str, nombre_artista: str) -> list[dict]:
    """Obtiene top tracks via search filtrado por artist_id."""
    tracks = []
    for query in [f'artist:"{nombre_artista}"', f"artist:{nombre_artista}"]:
        for limit in [10, 5]:
            res = safe_call(sp, sp.search, q=query, type="track", limit=limit, market="ES")
            if res:
                tracks = [
                    t for t in res.get("tracks", {}).get("items", [])
                    if any(a["id"] == artist_id for a in t.get("artists", []))
                ]
                if tracks:
                    break
        if tracks:
            break
    return [
        {
            "nombre_artista": nombre_artista,
            "artist_id":      artist_id,
            "track_id":       t.get("id"),
            "track_name":     t.get("name"),
            "album_name":     t.get("album", {}).get("name"),
            "release_date":   t.get("album", {}).get("release_date"),
            "duration_ms":    t.get("duration_ms"),
            "explicit":       t.get("explicit"),
            "num_artistas":   len(t.get("artists", [])),
        }
        for t in tracks[:10]
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrae features de Spotify por artista")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en los CSVs")
    args = parser.parse_args()

    from config.registry import cargar as cargar_registry
    df_reg = cargar_registry()
    df_ids = (
        df_reg[df_reg["spotify_id"].notna()][["nombre_canonico", "spotify_id", "spotify_nombre"]]
        .rename(columns={
            "nombre_canonico": "nombre_buscado",
            "spotify_id":      "artist_id",
            "spotify_nombre":  "nombre_spotify",
        })
        .reset_index(drop=True)
    )
    print(f"{len(df_ids)} artistas con Spotify ID en el registry")

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))
    print("Cliente Spotify inicializado\n")

    df_disc_existe   = pd.read_csv(DISCOGRAFIA_CSV) if DISCOGRAFIA_CSV.exists() else pd.DataFrame()
    df_tracks_existe = pd.read_csv(TOP_TRACKS_CSV)  if TOP_TRACKS_CSV.exists()  else pd.DataFrame()

    if args.force and not df_disc_existe.empty:
        force_ids = set(
            df_ids.loc[df_ids["nombre_buscado"].str.lower().isin(
                {a.lower() for a in args.force}), "artist_id"]
        )
        df_disc_existe = df_disc_existe[~df_disc_existe["artist_id"].isin(force_ids)]
        df_disc_existe.to_csv(DISCOGRAFIA_CSV, index=False, encoding="utf-8")
        if not df_tracks_existe.empty:
            df_tracks_existe = df_tracks_existe[~df_tracks_existe["artist_id"].isin(force_ids)]
            df_tracks_existe.to_csv(TOP_TRACKS_CSV, index=False, encoding="utf-8")
        print(f"Re-procesando: {args.force}")

    procesados    = set(df_disc_existe["artist_id"]) if not df_disc_existe.empty else set()
    nuevas_disc   = []
    nuevos_tracks = []
    total         = len(df_ids)

    for i, row in df_ids.iterrows():
        nombre         = row["nombre_buscado"]
        nombre_spotify = row.get("nombre_spotify") or nombre
        artist_id      = row["artist_id"]

        if artist_id in procesados:
            print(f"  [{i+1}/{total}] Omitido (ya existe): {nombre}")
            continue

        print(f"  [{i+1}/{total}] {nombre} ...", flush=True)

        disc = get_discografia(sp, artist_id, nombre_spotify)
        disc["nombre_buscado"] = nombre
        disc["artist_id"]      = artist_id
        nuevas_disc.append(disc)
        print(f"    discografia: {disc['num_total_releases']} releases | ultimo: {disc['ultimo_lanzamiento']}")
        time.sleep(DELAY)

        tracks = get_spotify_tracks(sp, artist_id, nombre_spotify)
        nuevos_tracks.extend(tracks)
        print(f"    top tracks: {len(tracks)} canciones")
        time.sleep(DELAY)

    cols_disc = ["nombre_buscado", "artist_id", "num_albums", "num_singles",
                 "num_eps", "num_total_releases", "primer_lanzamiento",
                 "ultimo_lanzamiento", "anos_activo", "releases_por_ano"]
    if nuevas_disc:
        df_disc_final = pd.concat(
            [df_disc_existe, pd.DataFrame(nuevas_disc)[cols_disc]],
            ignore_index=True,
        )
        df_disc_final.to_csv(DISCOGRAFIA_CSV, index=False, encoding="utf-8")
        print(f"\nOK {len(nuevas_disc)} artistas → {DISCOGRAFIA_CSV.name}")

    if nuevos_tracks:
        df_tracks_final = pd.concat(
            [df_tracks_existe, pd.DataFrame(nuevos_tracks)],
            ignore_index=True,
        )
        df_tracks_final.to_csv(TOP_TRACKS_CSV, index=False, encoding="utf-8")
        print(f"OK {len(nuevos_tracks)} tracks → {TOP_TRACKS_CSV.name}")

    print(f"\nRESUMEN: {len(nuevas_disc)} artistas procesados | {len(nuevos_tracks)} tracks extraídos")


if __name__ == "__main__":
    main()
