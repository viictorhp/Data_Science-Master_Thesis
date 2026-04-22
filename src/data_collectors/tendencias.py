"""
Recoge señales de tendencia/buzz actual por artista combinando dos fuentes:

  1. Google Trends (pytrends) — interés de búsqueda en España últimos 12 meses
  2. YouTube vídeos recientes (Data API v3) — vistas de los últimos 5 vídeos

Ambas fuentes son complementarias:
  - Google Trends: detecta artistas con masa crítica de búsquedas web (0 = sin presencia)
  - YouTube reciente: más sensible para artistas pequeños con vídeos virales puntuales

Genera: data/raw/tendencias.csv

Uso:
  python -m src.data_collectors.tendencias
  python -m src.data_collectors.tendencias --force RVFV ARON Maka
"""

import os
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pytrends.request import TrendReq

load_dotenv()

YOUTUBE_API_KEY  = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"

ARTISTS_FILE   = Path("artistas.txt")
YOUTUBE_CSV    = Path("data/raw/youtube_artistas.csv")
OUTPUT_CSV     = Path("data/raw/tendencias.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DELAY_TRENDS  = 3.0   # Google Trends es sensible al rate limit
DELAY_YOUTUBE = 0.5


# ── Google Trends ─────────────────────────────────────────────────────────────

def _init_pytrends() -> TrendReq:
    return TrendReq(hl="es-ES", tz=60, timeout=(10, 25))


def get_trends(pytrends: TrendReq, nombre: str) -> dict:
    """Interés medio y pico en España durante los últimos 12 meses (escala 0-100)."""
    for intento in range(3):
        try:
            pytrends.build_payload([nombre], geo="ES", timeframe="today 12-m")
            df = pytrends.interest_over_time()
            if df.empty or nombre not in df.columns:
                return {"gtrends_interes_medio": 0, "gtrends_pico_maximo": 0}
            serie = df[nombre]
            return {"gtrends_interes_medio": round(float(serie.mean()), 2)}
        except Exception as e:
            espera = 10 * (intento + 1)
            print(f"  [trends error intento {intento+1}/3] {e} — esperando {espera}s")
            time.sleep(espera)
    return {"gtrends_interes_medio": None}


# ── YouTube vídeos recientes ──────────────────────────────────────────────────

def _yt_get(endpoint: str, params: dict) -> dict | None:
    params["key"] = YOUTUBE_API_KEY
    try:
        r = requests.get(f"{YOUTUBE_BASE_URL}/{endpoint}", params=params, timeout=10)
        if r.status_code == 403:
            err = r.json().get("error", {})
            if any(e.get("reason") == "quotaExceeded" for e in err.get("errors", [])):
                raise RuntimeError("Quota diaria de YouTube agotada.")
            return None
        r.raise_for_status()
        return r.json()
    except RuntimeError:
        raise
    except Exception:
        return None


def get_youtube_reciente(channel_id: str) -> dict:
    """Vistas acumuladas y número de los últimos 5 vídeos del canal."""
    if not channel_id or pd.isna(channel_id):
        return {"yt_vistas_recientes": None, "yt_videos_recientes": None}

    # Buscar últimos 5 vídeos del canal ordenados por fecha
    data = _yt_get("search", {
        "part":       "id",
        "channelId":  channel_id,
        "order":      "date",
        "type":       "video",
        "maxResults": 5,
    })
    if not data:
        return {"yt_vistas_recientes": None, "yt_videos_recientes": None}

    video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item["id"]]
    if not video_ids:
        return {"yt_vistas_recientes": 0, "yt_videos_recientes": 0}

    # Obtener estadísticas de esos vídeos
    stats = _yt_get("videos", {
        "part": "statistics",
        "id":   ",".join(video_ids),
    })
    if not stats:
        return {"yt_vistas_recientes": None, "yt_videos_recientes": len(video_ids)}

    total_vistas = sum(
        int(item.get("statistics", {}).get("viewCount", 0))
        for item in stats.get("items", [])
    )
    return {
        "yt_vistas_recientes":  total_vistas,
        "yt_videos_recientes":  len(video_ids),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recoge tendencias: Google Trends + YouTube reciente")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    with open(ARTISTS_FILE, encoding="utf-8") as f:
        artistas = [l.strip() for l in f if l.strip()]

    # Cargar channel_ids desde youtube_artistas.csv
    channel_ids = {}
    if YOUTUBE_CSV.exists():
        df_yt = pd.read_csv(YOUTUBE_CSV)[["nombre_buscado", "channel_id"]].dropna(subset=["channel_id"])
        channel_ids = dict(zip(df_yt["nombre_buscado"].str.lower(), df_yt["channel_id"]))

    # Carga incremental
    df_existe = pd.read_csv(OUTPUT_CSV) if OUTPUT_CSV.exists() else pd.DataFrame()

    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        df_existe = df_existe[~df_existe["nombre_buscado"].str.lower().isin(force_lower)]
        df_existe.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Re-procesando: {args.force}")

    procesados = set(df_existe["nombre_buscado"].str.lower()) if not df_existe.empty else set()
    pendientes = [a for a in artistas if a.lower() not in procesados]

    print(f"Ya procesados: {len(procesados)} | Pendientes: {len(pendientes)}")
    if not pendientes:
        print("Todos los artistas ya están procesados.")
        return

    if not YOUTUBE_API_KEY:
        print("[aviso] YOUTUBE_API_KEY no encontrada — se omitirán datos de YouTube reciente")

    pytrends = _init_pytrends()
    cols = ["nombre_buscado", "gtrends_interes_medio",
            "yt_vistas_recientes", "yt_videos_recientes"]
    total = len(pendientes)
    guardados = 0

    for i, nombre in enumerate(pendientes, 1):
        print(f"[{i}/{total}] {nombre} ...", flush=True)

        # Google Trends
        trends = get_trends(pytrends, nombre)
        print(f"  trends → medio={trends['gtrends_interes_medio']}")
        time.sleep(DELAY_TRENDS)

        # YouTube reciente
        channel_id = channel_ids.get(nombre.lower())
        if YOUTUBE_API_KEY and channel_id:
            try:
                yt = get_youtube_reciente(channel_id)
                print(f"  youtube → vistas_recientes={yt['yt_vistas_recientes']} videos={yt['yt_videos_recientes']}")
            except RuntimeError as e:
                print(f"\n  [quota agotada] {e}")
                print(f"  Guardando progreso ({guardados} artistas ya en CSV)...")
                break
            time.sleep(DELAY_YOUTUBE)
        else:
            yt = {"yt_vistas_recientes": None, "yt_videos_recientes": None}
            if not channel_id:
                print("  youtube → sin channel_id")

        # Guardar tras cada artista
        fila = pd.DataFrame([{"nombre_buscado": nombre, **trends, **yt}]).reindex(columns=cols)
        df_existe = pd.concat([df_existe, fila], ignore_index=True)
        df_existe.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        guardados += 1

    print(f"\nOK {guardados} artistas nuevos → {OUTPUT_CSV}")
    print(f"RESUMEN: {len(df_existe)} artistas en CSV")


if __name__ == "__main__":
    main()
