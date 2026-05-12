"""
Obtiene datos de artistas desde Last.fm usando los nombres del registry.
Extrae: oyentes mensuales, scrobbles totales, géneros y biografía corta.

Usa lastfm_nombre del registry directamente (sin búsqueda por nombre), lo que
elimina los fallos de resolución que ocurren cuando el nombre canónico difiere
del nombre en Last.fm.

Genera: data/raw/lastfm_artistas.csv

Uso:
  python -m src.data_collectors.lastfmapi
  python -m src.data_collectors.lastfmapi --force RVFV Maka
"""

import os
import sys
import time
import difflib
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

API_KEY  = os.getenv("LASTFM_API_KEY")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"


class LastFMCollector:
    def __init__(self):
        if not API_KEY:
            raise ValueError("LASTFM_API_KEY no encontrada en .env")

    def _get(self, params: dict) -> dict:
        params.update({"api_key": API_KEY, "format": "json"})
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def obtener_info(self, nombre_lastfm: str) -> dict | None:
        """
        Obtiene datos completos de un artista usando su nombre exacto en Last.fm.
        No realiza búsqueda: usa el nombre tal como está en el registry.
        """
        try:
            data = self._get({"method": "artist.getInfo", "artist": nombre_lastfm})
            if "error" in data:
                print(f"  Last.fm error: {data.get('message')}")
                return None
            a     = data["artist"]
            tags  = [t["name"] for t in a.get("tags", {}).get("tag", [])]
            stats = a.get("stats", {})
            return {
                "nombre_lastfm": a.get("name"),
                "oyentes":       int(stats.get("listeners", 0)),
                "scrobbles":     int(stats.get("playcount", 0)),
                "generos":       ", ".join(tags) if tags else "Sin género",
                "lastfm_url":    a.get("url"),
                "mbid":          a.get("mbid", "") or None,
            }
        except Exception as e:
            print(f"  Error Last.fm para '{nombre_lastfm}': {e}")
            return None

    def buscar_artista(self, nombre: str, umbral: float = 0.6) -> dict | None:
        """
        Búsqueda por nombre con validación de similitud.
        Usado por scripts/00_resolver_identidades.py para poblar el registry.
        """
        try:
            busqueda = self._get({"method": "artist.search", "artist": nombre, "limit": 5})
            candidatos = busqueda.get("results", {}).get("artistmatches", {}).get("artist", [])
            if not candidatos:
                return None
            mejor = max(candidatos, key=lambda a: self._similitud(nombre, a["name"]))
            score = self._similitud(nombre, mejor["name"])
            if score < umbral:
                print(f"  Match rechazado: '{nombre}' → '{mejor['name']}' ({score:.0%})")
                return None
            data = self._get({"method": "artist.getInfo", "artist": mejor["name"]})
            if "error" in data:
                return None
            a    = data["artist"]
            tags = [t["name"] for t in a.get("tags", {}).get("tag", [])]
            return {
                "nombre_buscado": nombre,
                "nombre_lastfm":  a.get("name"),
                "oyentes":        int(a.get("stats", {}).get("listeners", 0)),
                "scrobbles":      int(a.get("stats", {}).get("playcount", 0)),
                "generos":        ", ".join(tags) if tags else "Sin género",
                "lastfm_url":     a.get("url"),
                "mbid":           a.get("mbid", "") or None,
                "match_score":    round(score, 2),
            }
        except Exception as e:
            print(f"  Error al buscar '{nombre}': {e}")
            return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recoge datos de artistas desde Last.fm")
    parser.add_argument("--force", nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    output = ROOT / "data" / "raw" / "lastfm_artistas.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    from config.registry import cargar as cargar_registry
    df_reg = cargar_registry()
    # Usar lastfm_nombre si está en el registry; si no, usar nombre_canonico como fallback
    df_reg["_nombre_lf"] = df_reg["lastfm_nombre"].fillna(df_reg["nombre_canonico"])

    df_existe = pd.read_csv(output) if output.exists() else pd.DataFrame()

    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        df_existe = df_existe[~df_existe["nombre_buscado"].str.lower().isin(force_lower)]
        df_existe.to_csv(output, index=False, encoding="utf-8")
        print(f"Re-procesando: {args.force}")

    ya_procesados = set(df_existe["nombre_buscado"].str.lower()) if not df_existe.empty else set()
    pendientes = df_reg[~df_reg["nombre_canonico"].str.lower().isin(ya_procesados)]

    print(f"Registry: {len(df_reg)} artistas | Ya procesados: {len(ya_procesados)} | Pendientes: {len(pendientes)}")
    if pendientes.empty:
        print("No hay artistas nuevos que procesar.")
        return

    collector = LastFMCollector()
    nuevas_filas   = []
    no_encontrados = []
    total          = len(pendientes)

    for i, (_, row) in enumerate(pendientes.iterrows(), 1):
        nombre_can = row["nombre_canonico"]
        nombre_lf  = row["_nombre_lf"]
        print(f"[{i}/{total}] {nombre_can} (Last.fm: '{nombre_lf}')")

        datos = None
        for intento in range(3):
            try:
                datos = collector.obtener_info(nombre_lf)
                break
            except Exception as e:
                espera = 10 * (intento + 1)
                print(f"  Error de conexión (intento {intento+1}/3), reintentando en {espera}s: {e}")
                time.sleep(espera)

        if datos:
            fila = {"nombre_buscado": nombre_can, **datos}
            print(f"  {datos['nombre_lastfm']}: {datos['oyentes']:,} oyentes | {datos['generos']}")
        else:
            fila = {"nombre_buscado": nombre_can}
            no_encontrados.append(nombre_can)
            print(f"  No encontrado")

        nuevas_filas.append(fila)
        time.sleep(0.5)

    cols = ["nombre_buscado", "nombre_lastfm", "oyentes", "scrobbles", "generos", "lastfm_url", "mbid"]
    df_nuevo  = pd.DataFrame(nuevas_filas).reindex(columns=cols)
    df_final  = pd.concat([df_existe, df_nuevo], ignore_index=True)
    df_final.to_csv(output, index=False, encoding="utf-8")
    print(f"\nOK {len(nuevas_filas)} artistas → {output.name}")

    if no_encontrados:
        print(f"Sin datos ({len(no_encontrados)}): {', '.join(no_encontrados)}")
        print("Revisa si el nombre en el registry (lastfm_nombre) es correcto.")


if __name__ == "__main__":
    main()
