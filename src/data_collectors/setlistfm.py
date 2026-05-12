"""
Obtiene el historial de conciertos de cada artista desde setlist.fm.
Usa el MBID (sl_mbid) o el nombre (sl_nombre) del registry directamente,
eliminando la dependencia de búsquedas por nombre y de lastfm_artistas.csv.

Genera: data/raw/setlistfm_conciertos.csv

Uso:
  python -m src.data_collectors.setlistfm
  python -m src.data_collectors.setlistfm --force Dano BEJO
"""

import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
API_KEY  = os.getenv("SETLIST_FM_API_KEY")
BASE_URL = "https://api.setlist.fm/rest/1.0"
HEADERS  = {"x-api-key": API_KEY or "", "Accept": "application/json"}

SETLISTS_CSV = ROOT / "data" / "raw" / "setlistfm_conciertos.csv"
SETLISTS_CSV.parent.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.6


def _similitud(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def buscar_mbid_por_nombre(nombre: str) -> tuple[str, str] | None:
    """
    Busca el MBID en setlist.fm por nombre cuando no hay MBID en el registry.
    Devuelve (mbid, nombre_sf) o None si no supera el umbral.
    """
    r = requests.get(
        f"{BASE_URL}/search/artists",
        headers=HEADERS,
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
    score = _similitud(nombre, nombre_sf)
    if score < SIMILARITY_THRESHOLD:
        print(f"  [match rechazado: '{nombre_sf}' score={score:.2f}]", end=" ")
        return None
    return a.get("mbid"), nombre_sf


def obtener_setlists(mbid: str, max_paginas: int = 5) -> list:
    todos = []
    for p in range(1, max_paginas + 1):
        r = requests.get(
            f"{BASE_URL}/artist/{mbid}/setlists",
            headers=HEADERS,
            params={"p": p},
            timeout=10,
        )
        if r.status_code != 200:
            break
        data     = r.json()
        setlists = data.get("setlist", [])
        if not setlists:
            break
        todos.extend(setlists)
        if len(todos) >= int(data.get("total", 0)):
            break
        time.sleep(1.0)
    return todos


def parsear_setlist(s: dict, mbid: str, nombre_canonico: str) -> dict:
    venue  = s.get("venue", {})
    ciudad = venue.get("city", {})
    sets   = s.get("sets", {}).get("set", [])

    total_canciones   = sum(len(st.get("song", [])) for st in sets)
    tiene_encore      = any(st.get("encore") for st in sets)
    nombres_canciones = [
        song.get("name", "")
        for st in sets for song in st.get("song", [])
    ]
    return {
        "nombre":          nombre_canonico,
        "setlistfm_mbid":  mbid,
        "setlist_id":      s.get("id"),
        "fecha":           s.get("eventDate"),
        "venue_id":        venue.get("id"),
        "venue_nombre":    venue.get("name"),
        "ciudad":          ciudad.get("name"),
        "pais":            ciudad.get("country", {}).get("name"),
        "coordenadas_lat": ciudad.get("coords", {}).get("lat"),
        "coordenadas_lon": ciudad.get("coords", {}).get("long"),
        "tour":            s.get("tour", {}).get("name") if s.get("tour") else None,
        "num_canciones":   total_canciones,
        "tiene_encore":    tiene_encore,
        "canciones":       " | ".join(nombres_canciones),
        "url_setlist":     s.get("url"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        raise EnvironmentError("SETLIST_FM_API_KEY no encontrada en .env")

    parser = argparse.ArgumentParser(description="Recoge conciertos de setlist.fm usando el registry")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    from config.registry import cargar as cargar_registry
    df_reg = cargar_registry()

    df_existe          = pd.read_csv(SETLISTS_CSV) if SETLISTS_CSV.exists() else pd.DataFrame()
    nombres_procesados = set(df_existe["nombre"].str.lower()) if not df_existe.empty else set()
    mbids_procesados   = set(df_existe["setlistfm_mbid"].dropna().astype(str)) if not df_existe.empty else set()

    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        df_existe = df_existe[~df_existe["nombre"].str.lower().isin(force_lower)].reset_index(drop=True)
        df_existe.to_csv(SETLISTS_CSV, index=False, encoding="utf-8")
        nombres_procesados = set(df_existe["nombre"].str.lower())
        mbids_procesados   = set(df_existe["setlistfm_mbid"].dropna().astype(str))
        print(f"Re-procesando: {args.force}")

    print(f"Registry: {len(df_reg)} artistas | Ya en CSV: {len(nombres_procesados)}")

    nuevas_filas   = []
    no_encontrados = []
    omitidos       = 0

    for _, row in df_reg.iterrows():
        nombre_can = row["nombre_canonico"]
        sl_mbid    = row.get("sl_mbid")
        sl_nombre  = row.get("sl_nombre") or nombre_can

        if nombre_can.lower() in nombres_procesados:
            omitidos += 1
            continue

        print(f"  Buscando: {nombre_can} ...", end=" ", flush=True)

        # Preferir MBID del registry; si no hay, buscar por nombre
        mbid = sl_mbid if pd.notna(sl_mbid) and str(sl_mbid).strip() else None

        if mbid and str(mbid) in mbids_procesados:
            print(f"omitido (MBID ya en CSV)")
            omitidos += 1
            continue

        if not mbid:
            resultado = buscar_mbid_por_nombre(str(sl_nombre))
            if not resultado:
                print("x no encontrado / match rechazado")
                no_encontrados.append(nombre_can)
                nuevas_filas.append({"nombre": nombre_can})
                continue
            mbid, nombre_sf = resultado
            if str(mbid) in mbids_procesados:
                print(f"omitido (MBID ya en CSV: '{nombre_sf}')")
                omitidos += 1
                continue
            print(f"ok MBID: {mbid} (nombre SF: '{nombre_sf}')")
        else:
            print(f"ok MBID desde registry: {mbid}")

        setlists = obtener_setlists(str(mbid), max_paginas=5)
        print(f"    → {len(setlists)} conciertos")

        if not setlists:
            nuevas_filas.append({"nombre": nombre_can, "setlistfm_mbid": mbid})
        else:
            for s in setlists:
                nuevas_filas.append(parsear_setlist(s, str(mbid), nombre_can))

        time.sleep(1.5)

    if nuevas_filas:
        df_nuevo = pd.DataFrame(nuevas_filas)
        df_final = pd.concat([df_existe, df_nuevo], ignore_index=True)
        df_final.to_csv(SETLISTS_CSV, index=False, encoding="utf-8")
        print(f"\nOK {len(nuevas_filas)} filas → {SETLISTS_CSV.name}")

    print(f"\nRESUMEN: {len(df_reg)} artistas | Omitidos: {omitidos} | No encontrados: {len(no_encontrados)}")
    if no_encontrados:
        print(f"Sin datos: {', '.join(no_encontrados)}")
        print("Revisa sl_nombre en el registry o añade sl_mbid manualmente.")


if __name__ == "__main__":
    main()
