import requests
import pandas as pd
import time
import os
from difflib import SequenceMatcher
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("SETLIST_FM_API_KEY")
if not API_KEY:
    raise EnvironmentError("SETLIST_FM_API_KEY no encontrada. Revisa tu .env")

BASE_URL = "https://api.setlist.fm/rest/1.0"
HEADERS = {
    "x-api-key": API_KEY,
    "Accept":    "application/json"
}

ARTISTS_FILE = Path("artistas.txt")
LASTFM_CSV   = Path("data/raw/lastfm_artistas.csv")
SETLISTS_CSV = Path("data/raw/setlistfm_conciertos.csv")
SETLISTS_CSV.parent.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.6   # mínimo para aceptar un match por nombre


def similitud(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def buscar_artista_por_nombre(nombre):
    """Busca en setlist.fm por nombre y valida similitud. Devuelve (mbid, nombre_sf) o None."""
    r = requests.get(
        f"{BASE_URL}/search/artists",
        headers=HEADERS,
        params={"artistName": nombre, "p": 1, "sort": "relevance"}
    )
    if r.status_code != 200:
        return None
    items = r.json().get("artist", [])
    if not items:
        return None
    a = items[0]
    nombre_encontrado = a.get("name", "")
    score = similitud(nombre, nombre_encontrado)
    if score < SIMILARITY_THRESHOLD:
        print(f"[match rechazado: '{nombre_encontrado}' score={score:.2f}]", end=" ")
        return None
    return a.get("mbid"), nombre_encontrado


def obtener_setlists(mbid, max_paginas=5):
    todos = []
    for p in range(1, max_paginas + 1):
        r = requests.get(
            f"{BASE_URL}/artist/{mbid}/setlists",
            headers=HEADERS,
            params={"p": p}
        )
        if r.status_code != 200:
            break
        data     = r.json()
        setlists = data.get("setlist", [])
        if not setlists:
            break
        todos.extend(setlists)
        total = int(data.get("total", 0))
        if len(todos) >= total:
            break
        time.sleep(1.0)
    return todos


def parsear_setlist(s, mbid, nombre):
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
        "nombre":          nombre,
        "setlistfm_mbid":  mbid,
        "setlist_id":      s.get("id"),
        "fecha":           s.get("eventDate"),        # dd-MM-yyyy
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


# ── Cargar MBIDs conocidos desde lastfm_artistas.csv ────────────────────────
mbids_conocidos = {}   # nombre_buscado.lower() → mbid
if LASTFM_CSV.exists():
    df_lastfm = pd.read_csv(LASTFM_CSV)
    for _, row in df_lastfm.iterrows():
        mbid = row.get("mbid")
        if pd.notna(mbid) and str(mbid).strip():
            mbids_conocidos[str(row["nombre_buscado"]).lower().strip()] = str(mbid).strip()

print(f"MBIDs conocidos desde Last.fm: {len(mbids_conocidos)}")

# ── Cargar ya procesados (incremental) ──────────────────────────────────────
df_existe          = pd.read_csv(SETLISTS_CSV) if SETLISTS_CSV.exists() else pd.DataFrame()
nombres_procesados = set(df_existe["nombre"].str.lower()) if not df_existe.empty else set()

with open(ARTISTS_FILE, encoding="utf-8") as f:
    nombres = [l.strip() for l in f if l.strip()]

nuevas_filas   = []
omitidos       = []
no_encontrados = []
matches_rechazados = []

# ── Bucle principal ──────────────────────────────────────────────────────────
for nombre in nombres:
    if nombre.lower() in nombres_procesados:
        print(f"  -> Omitido (ya existe): {nombre}")
        omitidos.append(nombre)
        continue

    print(f"  Buscando: {nombre} ...", end=" ")

    # 1. Intentar con MBID conocido de Last.fm (más fiable)
    mbid = mbids_conocidos.get(nombre.lower().strip())
    if mbid:
        print(f"ok  MBID desde Last.fm: {mbid}")
        nombre_sf = nombre   # ya sabemos quién es
    else:
        # 2. Buscar por nombre con validación de similitud
        resultado = buscar_artista_por_nombre(nombre)
        if not resultado:
            print("x no encontrado / match rechazado")
            no_encontrados.append(nombre)
            nuevas_filas.append({"nombre": nombre})
            continue
        mbid, nombre_sf = resultado
        print(f"ok  MBID: {mbid} (nombre: '{nombre_sf}')")

    setlists = obtener_setlists(mbid, max_paginas=5)
    print(f"    -> {len(setlists)} conciertos encontrados")

    if not setlists:
        nuevas_filas.append({"nombre": nombre_sf, "setlistfm_mbid": mbid})
    else:
        for s in setlists:
            nuevas_filas.append(parsear_setlist(s, mbid, nombre_sf))

    time.sleep(1.5)

# ── Guardar CSV ──────────────────────────────────────────────────────────────
if nuevas_filas:
    df_nuevo = pd.DataFrame(nuevas_filas)
    df_final = pd.concat([df_existe, df_nuevo], ignore_index=True)
    df_final.to_csv(SETLISTS_CSV, index=False, encoding="utf-8")
    print(f"\nOK {len(nuevas_filas)} filas -> {SETLISTS_CSV}")
else:
    print("\nNada nuevo que guardar.")

# ── Resumen ──────────────────────────────────────────────────────────────────
print(f"\nRESUMEN:")
print(f"   Total artistas en lista:   {len(nombres)}")
print(f"   Ya procesados (omitidos):  {len(omitidos)}")
print(f"   No encontrados/rechazados: {len(no_encontrados)}")
print(f"   Filas añadidas:            {len(nuevas_filas)}")

if no_encontrados:
    print(f"\n   No encontrados / match rechazado en setlist.fm:")
    for a in no_encontrados:
        print(f"      - {a}")
