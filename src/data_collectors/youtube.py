"""
Obtiene datos de canales de YouTube por artista usando la Data API v3.

Aporta señales de presencia digital que no están en ninguna otra fuente:
  - suscriptores:    tamaño de la audiencia fiel en YouTube
  - vistas_totales:  popularidad acumulada del canal
  - num_videos:      actividad/consistencia de contenido
  - fecha_creacion:  antigüedad del canal (proxy de trayectoria)
  - pais_canal:      país del canal cuando está disponible

QUOTA: La API tiene 10.000 unidades/día. Cada artista consume ~100 unidades
(búsqueda) + 1 (estadísticas). Con 125 artistas son ~12.625 unidades.
El procesado incremental omite artistas ya guardados automáticamente.

Uso:
  python -m src.data_collectors.youtube                  # todos los pendientes
  python -m src.data_collectors.youtube --batch-size 90  # solo 90 artistas
  python -m src.data_collectors.youtube --force Dano BEJO Choclock
      → borra esas filas del CSV y las re-procesa
"""

import os
import time
import difflib
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.googleapis.com/youtube/v3"
API_KEY  = os.getenv("YOUTUBE_API_KEY")

# Palabras en descripción/título que descalifican un canal como no-musical
_KEYWORDS_EXCLUIR = {
    "minecraft", "bedwars", "skywars", "gaming", "gameplay", "pvp",
    "meme", "memes", "karaoke", "instrumental",
    "fan account", "cuenta de fans", "animar y apoyar",
    "canal de fans", "bachata mix", "mix de bachata",
}

# "- Topic" → canales auto-generados por YouTube Music; menos útiles que el canal real
_TOPIC_SUFFIX = "- topic"


class YouTubeCollector:
    def __init__(self, api_key: str = API_KEY, delay: float = 0.5):
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY no encontrada en .env")
        self.api_key = api_key
        self.delay   = delay
        self.session = requests.Session()
        print("✓ Cliente YouTube inicializado")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    def _get(self, endpoint: str, params: dict, retries: int = 3):
        url = f"{BASE_URL}/{endpoint}"
        params["key"] = self.api_key

        for intento in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 403:
                    data = r.json()
                    if any(
                        e.get("reason") == "quotaExceeded"
                        for e in data.get("error", {}).get("errors", [])
                    ):
                        raise RuntimeError(
                            "Quota diaria agotada. Vuelve a ejecutar mañana."
                        )
                    print(f"  [403] {data.get('error', {}).get('message')}")
                    return None
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except RuntimeError:
                raise
            except requests.exceptions.RequestException as e:
                if intento < retries - 1:
                    time.sleep(5 * (intento + 1))
                else:
                    print(f"  Error tras {retries} intentos: {e}")
                    return None

    # ── Validación de canal ───────────────────────────────────────────────────

    @staticmethod
    def _es_no_musical(titulo: str, descripcion: str) -> bool:
        """True si el canal claramente no es de música (gaming, memes, karaoke...)."""
        texto = (titulo + " " + descripcion).lower()
        return any(kw in texto for kw in _KEYWORDS_EXCLUIR)

    @staticmethod
    def _es_muy_pequeno(subs: int | None, vistas: int | None) -> bool:
        """Canal con <50 subs y <5.000 vistas es casi siempre un match erróneo."""
        if subs is None and vistas is None:
            return False
        s = subs or 0
        v = vistas or 0
        return s < 50 and v < 5_000

    # ── Puntuación de candidatos ──────────────────────────────────────────────

    def _score(self, nombre: str, item: dict) -> float:
        """
        Puntuación combinada para elegir el canal más probable.
        Factores:
          - Similitud de nombre (principal)
          - Penalización por canal Topic auto-generado
          - Bonus por palabras que indican canal oficial de música
        """
        titulo = item["snippet"].get("title", "")
        titulo_limpio = (
            titulo.lower()
            .replace("oficial", "").replace("official", "")
            .replace("music", "").replace("música", "")
            .replace("vevo", "").replace(_TOPIC_SUFFIX, "")
            .strip()
        )
        sim = self._similitud(nombre, titulo_limpio)

        # Penalización por canal Topic (auto-generado, sin actividad real)
        if _TOPIC_SUFFIX in titulo.lower():
            sim *= 0.80

        # Penalización leve por fan account (poca probabilidad de ser oficial)
        desc = item["snippet"].get("description", "").lower()
        if "fan" in desc and ("cuenta" in desc or "account" in desc):
            sim *= 0.70

        return sim

    # ── Búsqueda de canal ─────────────────────────────────────────────────────

    def buscar_canal(self, nombre: str, umbral: float = 0.55) -> dict | None:
        """
        Busca el canal oficial del artista en YouTube con dos estrategias:

        1. Búsqueda filtrada por topicId de música (/m/04rlf) — filtra a canales
           de música directamente desde la API.
        2. Fallback genérico si la búsqueda musical no produce resultados
           aceptables — aplica filtros de descripción y scoring mejorado.

        Solo se consume la 2.ª búsqueda si la 1.ª falla o devuelve score bajo,
        minimizando el impacto en la quota diaria.
        """
        candidatos = []

        # ── Estrategia 1: topicId música ──────────────────────────────────────
        data = self._get("search", {
            "part":            "snippet",
            "q":               nombre,
            "type":            "channel",
            "topicId":         "/m/04rlf",
            "maxResults":      8,
            "relevanceLanguage": "es",
        })
        if data:
            for item in data.get("items", []):
                titulo = item["snippet"].get("title", "")
                desc   = item["snippet"].get("description", "")
                if not self._es_no_musical(titulo, desc):
                    candidatos.append(item)

        # Buscar el mejor entre los candidatos musicales
        mejor, mejor_score = None, 0.0
        for item in candidatos:
            s = self._score(nombre, item)
            if s > mejor_score:
                mejor_score, mejor = s, item

        # Si el resultado ya es suficientemente bueno, devolver
        if mejor and mejor_score >= umbral:
            return {
                "channel_id":   mejor["snippet"]["channelId"],
                "nombre_canal": mejor["snippet"]["title"],
                "match_score":  round(mejor_score, 2),
            }

        # ── Estrategia 2: búsqueda genérica con términos de rap/música ────────
        data2 = self._get("search", {
            "part":            "snippet",
            "q":               f"{nombre} rapper artista música",
            "type":            "channel",
            "maxResults":      8,
            "relevanceLanguage": "es",
        })
        if data2:
            for item in data2.get("items", []):
                titulo = item["snippet"].get("title", "")
                desc   = item["snippet"].get("description", "")
                if not self._es_no_musical(titulo, desc):
                    s = self._score(nombre, item)
                    if s > mejor_score:
                        mejor_score, mejor = s, item

        if not mejor or mejor_score < umbral:
            titulo_rej = mejor["snippet"].get("title", "?") if mejor else "—"
            print(f"  ⚠  Match rechazado: '{nombre}' → '{titulo_rej}' ({mejor_score:.0%})")
            return None

        return {
            "channel_id":   mejor["snippet"]["channelId"],
            "nombre_canal": mejor["snippet"]["title"],
            "match_score":  round(mejor_score, 2),
        }

    # ── Estadísticas del canal ────────────────────────────────────────────────

    def obtener_stats(self, channel_id: str) -> dict:
        data = self._get("channels", {
            "part": "statistics,snippet",
            "id":   channel_id,
        })
        if not data:
            return {}

        items = data.get("items", [])
        if not items:
            return {}

        stats   = items[0].get("statistics", {})
        snippet = items[0].get("snippet", {})

        subs   = int(stats["subscriberCount"]) if not stats.get("hiddenSubscriberCount") and "subscriberCount" in stats else None
        vistas = int(stats.get("viewCount", 0)) or None
        videos = int(stats.get("videoCount", 0)) or None

        return {
            "suscriptores":      subs,
            "vistas_totales":    vistas,
            "num_videos":        videos,
            "fecha_creacion":    snippet.get("publishedAt", "")[:10] or None,
            "pais_canal":        snippet.get("country"),
            "descripcion_canal": (snippet.get("description") or "")[:200] or None,
            "_muy_pequeno":      self._es_muy_pequeno(subs, vistas),
        }

    # ── Procesado de lista ────────────────────────────────────────────────────

    def procesar_lista(self, lista_artistas: list, procesados: set) -> list:
        nuevas_filas   = []
        no_encontrados = []
        sospechosos    = []
        total = len(lista_artistas)

        print(f"\nProcesando {total} artistas...\n")

        for i, nombre in enumerate(lista_artistas, 1):
            prefijo = f"[{i}/{total}]"

            if nombre.lower() in procesados:
                print(f"{prefijo} Omitido (ya existe): {nombre}")
                continue

            print(f"{prefijo} Buscando: {nombre} ...", end=" ", flush=True)

            try:
                canal = self.buscar_canal(nombre)
            except RuntimeError as e:
                print(f"\n{e}")
                print(f"  Guardando progreso ({len(nuevas_filas)} artistas)...")
                break

            if not canal:
                print("no encontrado")
                no_encontrados.append(nombre)
                nuevas_filas.append({"nombre_buscado": nombre})
                time.sleep(self.delay)
                continue

            stats = self.obtener_stats(canal["channel_id"])
            muy_pequeno = stats.pop("_muy_pequeno", False)

            fila = {
                "nombre_buscado": nombre,
                "nombre_canal":   canal["nombre_canal"],
                "channel_id":     canal["channel_id"],
                "match_score":    canal["match_score"],
                **stats,
            }
            nuevas_filas.append(fila)

            subs = f"{stats.get('suscriptores'):,}" if stats.get("suscriptores") else "ocultos"
            aviso = " ⚠ MUY PEQUEÑO" if muy_pequeno else ""
            print(
                f"✓ {canal['nombre_canal']} | "
                f"subs={subs} | score={canal['match_score']:.0%}"
                f"{aviso}"
            )
            if muy_pequeno:
                sospechosos.append(nombre)

            time.sleep(self.delay)

        print(f"\n✓ Procesados: {len(nuevas_filas)}")
        if no_encontrados:
            print(f"  No encontrados ({len(no_encontrados)}): {', '.join(no_encontrados)}")
        if sospechosos:
            print(f"  ⚠  Canal muy pequeño — revisar manualmente: {', '.join(sospechosos)}")

        return nuevas_filas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recoge datos de YouTube Data API v3")
    parser.add_argument("--artists",    default="artistas.txt",
                        help="Fichero con un artista por línea")
    parser.add_argument("--output",     default="data/raw/youtube_artistas.csv",
                        help="CSV de salida")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Artistas a procesar por ejecución (0=todos)")
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Segundos entre llamadas a la API")
    parser.add_argument("--force",      nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    if not os.path.exists(args.artists):
        print(f"No se encuentra: {args.artists}")
        return

    with open(args.artists, encoding="utf-8") as f:
        artistas = [l.strip() for l in f if l.strip()]
    print(f"{len(artistas)} artistas en {args.artists}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_existe = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    # ── Borrar filas de artistas forzados ────────────────────────────────────
    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        mask = df_existe["nombre_buscado"].str.lower().isin(force_lower)
        borrados = df_existe[mask]["nombre_buscado"].tolist()
        df_existe = df_existe[~mask].reset_index(drop=True)
        if borrados:
            df_existe.to_csv(out_path, index=False, encoding="utf-8")
            print(f"Re-procesando: {borrados}")

    procesados = (
        set(df_existe["nombre_buscado"].str.lower())
        if not df_existe.empty else set()
    )
    pendientes = [a for a in artistas if a.lower() not in procesados]

    print(f"  Ya procesados: {len(procesados)}")
    print(f"  Pendientes:    {len(pendientes)}")

    if not pendientes:
        print("Todos los artistas ya están procesados.")
        return

    lote = pendientes[:args.batch_size] if args.batch_size > 0 else pendientes
    quota_est = len(lote) * 102
    print(f"  Este lote: {len(lote)} artistas (~{quota_est} unidades de quota)\n")

    collector = YouTubeCollector(delay=args.delay)
    nuevas_filas = collector.procesar_lista(lote, procesados)

    cols = [
        "nombre_buscado", "nombre_canal", "channel_id", "match_score",
        "suscriptores", "vistas_totales", "num_videos",
        "fecha_creacion", "pais_canal", "descripcion_canal",
    ]
    if nuevas_filas:
        df_nuevo = pd.DataFrame(nuevas_filas).reindex(columns=cols)
        df_final = pd.concat([df_existe, df_nuevo], ignore_index=True)
        df_final.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n{len(nuevas_filas)} artistas → {out_path}")

    total_csv = len(df_existe) + len(nuevas_filas)
    print(f"\nRESUMEN: {total_csv}/{len(artistas)} artistas en CSV")


if __name__ == "__main__":
    main()
