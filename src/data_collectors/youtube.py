"""
Obtiene datos de canales de YouTube por artista usando la Data API v3.

Aporta señales de presencia digital que no están en ninguna otra fuente:
  - suscriptores:    tamaño de la audiencia fiel en YouTube
  - vistas_totales:  popularidad acumulada del canal
  - num_videos:      actividad/consistencia de contenido
  - fecha_creacion:  antigüedad del canal (proxy de trayectoria)
  - pais_canal:      país del canal cuando está disponible

QUOTA: La API tiene 10.000 unidades/día.
  - Búsqueda (usada solo en 00_resolver_identidades.py): ~100 unidades/artista
  - Estadísticas de canal conocido (main de este script): 1 unidad/artista

El main usa channel_id del registry directamente — no hace búsquedas.
buscar_canal() solo lo invoca el resolver para nuevos artistas.

Uso:
  python -m src.data_collectors.youtube
  python -m src.data_collectors.youtube --batch-size 90
  python -m src.data_collectors.youtube --force Dano BEJO Choclock
"""

import os
import sys
import time
import difflib
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT     = Path(__file__).parent.parent.parent
BASE_URL = "https://www.googleapis.com/youtube/v3"
API_KEY  = os.getenv("YOUTUBE_API_KEY")

sys.path.insert(0, str(ROOT))

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
        1. Búsqueda filtrada por topicId de música (/m/04rlf).
        2. Fallback genérico si la búsqueda musical no produce resultados aceptables.

        Usado por scripts/00_resolver_identidades.py para nuevos artistas.
        Las correcciones manuales se gestionan en el registry (manual_override=True)
        y no necesitan pasar por este método.
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """
    Recolecta estadísticas de YouTube usando channel_id del registry.
    No realiza búsquedas (buscar_canal solo lo usa 00_resolver_identidades.py).
    Quota consumida: ~1 unidad/artista (solo channels.list con ID conocido).
    """
    parser = argparse.ArgumentParser(description="Recoge estadísticas de YouTube usando IDs del registry")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Artistas a procesar por ejecución (0=todos)")
    parser.add_argument("--delay",      type=float, default=0.5)
    parser.add_argument("--force",      nargs="+", metavar="ARTISTA",
                        help="Re-procesar estos artistas aunque ya existan en el CSV")
    args = parser.parse_args()

    from config.registry import cargar as cargar_registry
    df_reg = cargar_registry()
    # Solo artistas con channel_id resuelto
    df_con_canal = df_reg[df_reg["yt_channel_id"].notna()][["nombre_canonico", "yt_channel_id"]].copy()
    df_sin_canal = df_reg[df_reg["yt_channel_id"].isna()]["nombre_canonico"].tolist()

    out_path = ROOT / "data" / "raw" / "youtube_artistas.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_existe = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    if args.force and not df_existe.empty:
        force_lower = {a.lower() for a in args.force}
        df_existe = df_existe[~df_existe["nombre_buscado"].str.lower().isin(force_lower)].reset_index(drop=True)
        df_existe.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Re-procesando: {args.force}")

    procesados = set(df_existe["nombre_buscado"].str.lower()) if not df_existe.empty else set()
    pendientes_df = df_con_canal[~df_con_canal["nombre_canonico"].str.lower().isin(procesados)]

    print(f"Registry: {len(df_reg)} artistas | Con canal: {len(df_con_canal)} | Sin canal: {len(df_sin_canal)}")
    print(f"Ya procesados: {len(procesados)} | Pendientes: {len(pendientes_df)}")

    if pendientes_df.empty:
        print("Todos los artistas ya están procesados.")
        return

    lote = pendientes_df.head(args.batch_size) if args.batch_size > 0 else pendientes_df
    print(f"Este lote: {len(lote)} artistas (~{len(lote)} unidades de quota)\n")

    collector  = YouTubeCollector(delay=args.delay)
    nuevas_filas = []
    sospechosos  = []
    total        = len(lote)

    for i, (_, row) in enumerate(lote.iterrows(), 1):
        nombre     = row["nombre_canonico"]
        channel_id = row["yt_channel_id"]
        print(f"[{i}/{total}] {nombre} ({channel_id}) ...", end=" ", flush=True)

        try:
            stats = collector.obtener_stats(channel_id)
        except RuntimeError as e:
            print(f"\n{e}")
            print(f"Guardando progreso ({len(nuevas_filas)} artistas)...")
            break

        if not stats:
            print("sin datos")
            nuevas_filas.append({"nombre_buscado": nombre, "channel_id": channel_id})
            continue

        muy_pequeno = stats.pop("_muy_pequeno", False)
        fila = {"nombre_buscado": nombre, "channel_id": channel_id, **stats}
        nuevas_filas.append(fila)

        subs = f"{stats.get('suscriptores'):,}" if stats.get("suscriptores") else "ocultos"
        aviso = " CANAL MUY PEQUEÑO — revisar" if muy_pequeno else ""
        print(f"subs={subs}{aviso}")
        if muy_pequeno:
            sospechosos.append(nombre)

        time.sleep(args.delay)

    cols = [
        "nombre_buscado", "channel_id",
        "suscriptores", "vistas_totales", "num_videos",
        "fecha_creacion", "pais_canal", "descripcion_canal",
    ]
    if nuevas_filas:
        df_nuevo = pd.DataFrame(nuevas_filas).reindex(columns=cols)
        df_final = pd.concat([df_existe, df_nuevo], ignore_index=True)
        df_final.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n{len(nuevas_filas)} artistas → {out_path.name}")

    if sospechosos:
        print(f"Canal muy pequeño — revisar manualmente: {', '.join(sospechosos)}")
    if df_sin_canal:
        print(f"Sin canal en registry ({len(df_sin_canal)}): {', '.join(df_sin_canal[:5])}{'...' if len(df_sin_canal) > 5 else ''}")

    print(f"\nRESUMEN: {len(df_existe) + len(nuevas_filas)}/{len(df_reg)} artistas en CSV")


if __name__ == "__main__":
    main()
