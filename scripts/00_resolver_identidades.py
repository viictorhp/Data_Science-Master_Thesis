"""
Resuelve identidades de artistas nuevos en Spotify, Last.fm, YouTube y setlist.fm.
Actualiza config/artistas_registry.csv con los resultados.

El script añade SOLO la información que puede verificar con alta confianza:
  - Spotify:    score >= 80%
  - Last.fm:    score >= 80%
  - YouTube:    score >= 65%  (búsqueda menos precisa)
  - setlist.fm: score >= 80%

Si una plataforma no supera el umbral, el campo queda como NaN en el registry
y el artista aparece en pendientes_revision.csv para revisión manual.
Tras corregir manualmente, pon manual_override=True en esa fila.

Uso:
  python scripts/00_resolver_identidades.py                     # todos los pendientes
  python scripts/00_resolver_identidades.py --force RVFV Maka   # re-resolver artistas
  python scripts/00_resolver_identidades.py --input nuevos.txt  # fichero alternativo
"""

import argparse
import os
import sys
import time
import difflib
import requests
import pandas as pd
import spotipy
from pathlib import Path
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.registry import COLUMNS, cargar as cargar_registry, guardar as guardar_registry
from src.data_collectors.youtube import YouTubeCollector

PENDIENTES_CSV = ROOT / "pendientes_revision.csv"

UMBRAL_SPOTIFY = 0.80
UMBRAL_LASTFM  = 0.80
UMBRAL_YOUTUBE = 0.65   # YouTube es menos preciso; la validación manual es sencilla
UMBRAL_SL      = 0.80


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── Resolvers por plataforma ──────────────────────────────────────────────────

def _resolver_spotify(nombre: str, sp) -> dict | None:
    try:
        res = sp.search(q=f"artist:{nombre}", type="artist", limit=5)
        items = res["artists"]["items"]
        if not items:
            return None
        mejor = max(items, key=lambda a: _sim(nombre, a["name"]))
        score = _sim(nombre, mejor["name"])
        if score < UMBRAL_SPOTIFY:
            print(f"  [Spotify]    rechazado: '{mejor['name']}' ({score:.0%})")
            return None
        print(f"  [Spotify]    {mejor['name']} ({score:.0%})")
        return {
            "spotify_id":     mejor["id"],
            "spotify_nombre": mejor["name"],
            "spotify_score":  round(score, 2),
        }
    except Exception as e:
        print(f"  [Spotify]    error: {e}")
        return None


def _resolver_lastfm(nombre: str) -> dict | None:
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        print("  [Last.fm]    LASTFM_API_KEY no configurada")
        return None
    try:
        r = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={"method": "artist.search", "artist": nombre, "limit": 5,
                    "api_key": api_key, "format": "json"},
            timeout=10,
        )
        candidatos = r.json().get("results", {}).get("artistmatches", {}).get("artist", [])
        if not candidatos:
            return None
        mejor = max(candidatos, key=lambda a: _sim(nombre, a["name"]))
        score = _sim(nombre, mejor["name"])
        if score < UMBRAL_LASTFM:
            print(f"  [Last.fm]    rechazado: '{mejor['name']}' ({score:.0%})")
            return None
        # Obtener MBID real via artist.getInfo
        info = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={"method": "artist.getInfo", "artist": mejor["name"],
                    "api_key": api_key, "format": "json"},
            timeout=10,
        ).json()
        nombre_lf = info.get("artist", {}).get("name", mejor["name"])
        mbid = info.get("artist", {}).get("mbid") or None
        print(f"  [Last.fm]    {nombre_lf} ({score:.0%})")
        return {"lastfm_nombre": nombre_lf, "lastfm_mbid": mbid, "lastfm_score": round(score, 2)}
    except Exception as e:
        print(f"  [Last.fm]    error: {e}")
        return None


def _resolver_youtube(nombre: str, yt_col: YouTubeCollector) -> dict | None:
    try:
        canal = yt_col.buscar_canal(nombre, umbral=UMBRAL_YOUTUBE)
        if not canal:
            return None
        print(f"  [YouTube]    {canal['nombre_canal']} ({canal['match_score']:.0%})")
        return {
            "yt_channel_id":  canal["channel_id"],
            "yt_nombre_canal": canal["nombre_canal"],
            "yt_score":       canal["match_score"],
        }
    except Exception as e:
        print(f"  [YouTube]    error: {e}")
        return None


def _resolver_setlistfm(nombre: str) -> dict | None:
    api_key = os.getenv("SETLIST_FM_API_KEY")
    if not api_key:
        print("  [setlist.fm] SETLIST_FM_API_KEY no configurada")
        return None
    try:
        r = requests.get(
            "https://api.setlist.fm/rest/1.0/search/artists",
            headers={"x-api-key": api_key, "Accept": "application/json"},
            params={"artistName": nombre, "p": 1, "sort": "relevance"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        items = r.json().get("artist", [])
        if not items:
            return None
        a = items[0]
        nombre_sf = a.get("name", "")
        score = _sim(nombre, nombre_sf)
        if score < UMBRAL_SL:
            print(f"  [setlist.fm] rechazado: '{nombre_sf}' ({score:.0%})")
            return None
        print(f"  [setlist.fm] {nombre_sf} ({score:.0%})")
        return {"sl_nombre": nombre_sf, "sl_mbid": a.get("mbid") or None, "sl_score": round(score, 2)}
    except Exception as e:
        print(f"  [setlist.fm] error: {e}")
        return None


# ── Resolver principal ────────────────────────────────────────────────────────

def resolver_artista(nombre: str, sp, yt_col: YouTubeCollector) -> tuple[dict, list[str]]:
    """
    Resuelve identidades en las 4 plataformas.
    Devuelve (entrada_registry, plataformas_sin_resolver).
    Los campos sin resolver quedan ausentes de la entrada (→ NaN en el registry).
    """
    entrada = {"nombre_canonico": nombre, "manual_override": False, "notas": ""}
    sin_resolver = []

    for plataforma, fn in [
        ("spotify",   lambda: _resolver_spotify(nombre, sp)),
        ("lastfm",    lambda: _resolver_lastfm(nombre)),
        ("youtube",   lambda: _resolver_youtube(nombre, yt_col)),
        ("setlistfm", lambda: _resolver_setlistfm(nombre)),
    ]:
        resultado = fn()
        if resultado:
            entrada.update(resultado)
        else:
            sin_resolver.append(plataforma)
        time.sleep(0.5)

    return entrada, sin_resolver


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resuelve identidades de artistas en las 4 plataformas")
    parser.add_argument("--input", default="artistas.txt",
                        help="Fichero con un artista por línea")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-resolver estos artistas (no sobreescribe manual_override=True)")
    args = parser.parse_args()

    with open(ROOT / args.input, encoding="utf-8") as f:
        artistas = [l.strip() for l in f if l.strip()]
    print(f"{len(artistas)} artistas en {args.input}\n")

    try:
        df_reg = cargar_registry()
    except FileNotFoundError:
        print("Registry no encontrado — creando desde cero.")
        print("Tip: si ya tienes CSVs recolectados, ejecuta antes scripts/migrar_registry.py\n")
        df_reg = pd.DataFrame(columns=COLUMNS)

    procesados = set(df_reg["nombre_canonico"].str.lower())

    if args.force:
        force_lower = {a.lower() for a in args.force}
        # Eliminar filas no manuales de los forzados para re-resolverlas
        mask_forzar = df_reg["nombre_canonico"].str.lower().isin(force_lower) & ~df_reg["manual_override"]
        df_reg = df_reg[~mask_forzar].reset_index(drop=True)
        procesados = set(df_reg["nombre_canonico"].str.lower())
        pendientes = [a for a in artistas if a.lower() in force_lower or a.lower() not in procesados]
    else:
        pendientes = [a for a in artistas if a.lower() not in procesados]

    if not pendientes:
        print("Todos los artistas ya están en el registry.")
        return

    print(f"{len(pendientes)} artistas a resolver\n")

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))
    yt_col = YouTubeCollector()

    nuevas_filas = []
    problemas    = []

    for i, nombre in enumerate(pendientes, 1):
        print(f"[{i}/{len(pendientes)}] {nombre}")
        entrada, sin_resolver = resolver_artista(nombre, sp, yt_col)
        nuevas_filas.append(entrada)
        if sin_resolver:
            problemas.append({
                "nombre":                nombre,
                "plataformas_pendientes": ", ".join(sin_resolver),
            })
        print()

    df_nuevas = pd.DataFrame(nuevas_filas).reindex(columns=COLUMNS)
    df_reg    = pd.concat([df_reg, df_nuevas], ignore_index=True)
    guardar_registry(df_reg)
    print(f"Registry actualizado: {len(df_reg)} artistas → config/artistas_registry.csv")

    if problemas:
        df_pend = pd.DataFrame(problemas)
        df_pend.to_csv(PENDIENTES_CSV, index=False, encoding="utf-8")
        print(f"\n{len(problemas)} artistas con plataformas sin resolver → {PENDIENTES_CSV.name}")
        print("Rellena los campos en config/artistas_registry.csv y pon manual_override=True.")
    else:
        print("Todas las plataformas resueltas correctamente.")


if __name__ == "__main__":
    main()
