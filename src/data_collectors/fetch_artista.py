"""
fetch_artista.py
================
Orquestador de recuperación automática de datos de un artista para predicción.

Estrategia:
  1. Buscar en Spotify (la más robusta) → obtener nombre oficial + spotify_id
  2. Desambiguar con Groq si hay candidatos muy similares (opcional)
  3. Obtener datos de Last.fm, YouTube, setlist.fm y Tendencias en paralelo
  4. Transformar el ResultadoFetch en el vector de kwargs para predecir()

Uso desde Streamlit:
    from src.data_collectors.fetch_artista import (
        buscar_candidatos_spotify, fetch_features_por_nombre, resultado_a_features
    )
"""

import difflib
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidatoSpotify:
    spotify_id: str
    nombre_spotify: str
    score: float
    generos: list
    imagen_url: Optional[str] = None


@dataclass
class ResultadoFetch:
    nombre_spotify: str
    spotify_id: str
    sp: dict = field(default_factory=dict)
    lfm: dict = field(default_factory=dict)
    yt: dict = field(default_factory=dict)
    sl: dict = field(default_factory=dict)
    trend: dict = field(default_factory=dict)
    # "ok" | "not_found" | "error" | "skipped"
    estado: dict = field(default_factory=dict)
    advertencias: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _normalizar_nombre(nombre: str) -> str:
    """Fallback para búsqueda: lowercase + sin tildes + sin caracteres especiales."""
    nombre = nombre.strip().lower()
    nfd = unicodedata.normalize("NFD", nombre)
    sin_tildes = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    limpio = re.sub(r"[^\w\s.\-']", "", sin_tildes)
    return re.sub(r"\s+", " ", limpio).strip()


# ---------------------------------------------------------------------------
# Paso 1: Búsqueda en Spotify
# ---------------------------------------------------------------------------

def buscar_candidatos_spotify(
    nombre: str,
    sp_client,
    limit: int = 5,
) -> list:
    """
    Devuelve hasta `limit` CandidatoSpotify ordenados por similitud difflib.
    Filtra candidatos con score < 0.35 (demasiado alejados del nombre buscado).
    """
    try:
        res = sp_client.search(q=f"artist:{nombre}", type="artist", limit=limit)
        items = res.get("artists", {}).get("items", [])
        if not items:
            return []

        candidatos = []
        for item in items:
            nombre_sp = item.get("name", "")
            score = _sim(nombre, nombre_sp)
            if score < 0.35:
                continue
            generos = item.get("genres", [])
            imagenes = item.get("images", [])
            imagen_url = imagenes[0]["url"] if imagenes else None
            candidatos.append(CandidatoSpotify(
                spotify_id=item["id"],
                nombre_spotify=nombre_sp,
                score=round(score, 3),
                generos=generos,
                imagen_url=imagen_url,
            ))

        candidatos.sort(key=lambda c: c.score, reverse=True)
        return candidatos

    except Exception as e:
        print(f"  [Spotify búsqueda] error: {e}")
        return []


# ---------------------------------------------------------------------------
# Desambiguación con Groq (solo cuando los scores son muy próximos)
# ---------------------------------------------------------------------------

def desambiguar_con_groq(
    nombre_usuario: str,
    candidatos: list,
    groq_api_key: str,
) -> Optional[CandidatoSpotify]:
    """
    Usa Llama 3.3 70B para elegir el candidato más probable en el contexto
    de rap/urbano español. Temperatura 0.0 para respuesta determinista.

    Solo se llama cuando los dos primeros candidatos tienen scores a ≤ 0.05 puntos.
    Si Groq falla, devuelve None (el caller cae al selector manual).
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        opciones = "\n".join(
            f"{i+1}. {c.nombre_spotify} — géneros: {', '.join(c.generos[:3]) or 'desconocido'}"
            for i, c in enumerate(candidatos[:4])
        )
        messages = [
            SystemMessage(content=(
                "Eres un experto en rap y música urbana española. "
                "Debes elegir qué artista de la lista corresponde al buscado. "
                "Responde SOLO con el número del candidato (1, 2, 3 o 4). "
                "Si ninguno parece ser rap/urbano español, elige el más relevante."
            )),
            HumanMessage(content=(
                f"El usuario busca al artista: '{nombre_usuario}'\n\n"
                f"Candidatos en Spotify:\n{opciones}\n\n"
                "¿Cuál es el correcto? Responde solo con el número."
            )),
        ]
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=groq_api_key,
        )
        respuesta = llm.invoke(messages).content.strip()
        primer_char = respuesta[0] if respuesta else ""
        if not primer_char.isdigit():
            print(f"  [Groq desambiguación] respuesta inesperada: {respuesta[:60]!r}")
            return None
        idx = int(primer_char) - 1
        if 0 <= idx < len(candidatos):
            return candidatos[idx]
        print(f"  [Groq desambiguación] índice fuera de rango: {idx} (hay {len(candidatos)} candidatos)")
    except Exception as e:
        print(f"  [Groq desambiguación] error: {e}")
    return None


# ---------------------------------------------------------------------------
# Fetchers por plataforma (privados)
# ---------------------------------------------------------------------------

def _fetch_spotify(sp_client, spotify_id: str, nombre_spotify: str) -> dict:
    from src.data_collectors.spotify_features import get_discografia, get_spotify_tracks

    disc = get_discografia(sp_client, spotify_id, nombre_spotify)
    tracks = get_spotify_tracks(sp_client, spotify_id, nombre_spotify)

    avg_duration = 210_000.0
    pct_explicit = 0.6
    pct_colabs = 0.3

    if tracks:
        durations = [t["duration_ms"] for t in tracks if t.get("duration_ms")]
        if durations:
            avg_duration = sum(durations) / len(durations)
        explicits = [t["explicit"] for t in tracks if t.get("explicit") is not None]
        if explicits:
            pct_explicit = sum(1 for e in explicits if e) / len(explicits)
        if tracks:
            pct_colabs = sum(1 for t in tracks if t.get("num_artistas", 1) > 1) / len(tracks)

    return {
        "sp_num_albums":     disc.get("num_albums", 0),
        "sp_num_singles":    disc.get("num_singles", 0),
        "sp_anos_activo":    disc.get("anos_activo") or 1.0,
        "sp_avg_duration_ms": avg_duration,
        "sp_pct_explicit":   pct_explicit,
        "sp_pct_colabs":     pct_colabs,
    }


def _fetch_lastfm(nombre: str, api_key: str) -> dict:
    from src.data_collectors.lastfmapi import LastFMCollector

    collector = LastFMCollector()
    datos = collector.buscar_artista(nombre, umbral=0.70)
    if not datos:
        nombre_norm = _normalizar_nombre(nombre)
        if nombre_norm != nombre.lower().strip():
            datos = collector.buscar_artista(nombre_norm, umbral=0.65)
    if not datos:
        return {}

    num_generos = 0
    generos_str = datos.get("generos", "")
    if generos_str and generos_str not in ("Sin género", ""):
        num_generos = len(generos_str.split(","))

    return {
        "lfm_oyentes":    int(datos.get("oyentes", 0)),
        "lfm_scrobbles":  int(datos.get("scrobbles", 0)),
        "lfm_num_generos": num_generos,
    }


def _fetch_youtube(nombre: str, api_key: str) -> dict:
    from src.data_collectors.youtube import YouTubeCollector

    collector = YouTubeCollector(api_key=api_key)
    canal = collector.buscar_canal(nombre, umbral=0.50)
    if not canal:
        nombre_norm = _normalizar_nombre(nombre)
        if nombre_norm != nombre.lower().strip():
            canal = collector.buscar_canal(nombre_norm, umbral=0.45)
    if not canal:
        return {}

    channel_id = canal["channel_id"]
    stats = collector.obtener_stats(channel_id)
    if not stats:
        return {"channel_id": channel_id}

    stats.pop("_muy_pequeno", None)
    return {
        "channel_id":       channel_id,
        "yt_suscriptores":  stats.get("suscriptores") or 0,
        "yt_vistas_totales": stats.get("vistas_totales") or 0,
        "yt_num_videos":    stats.get("num_videos") or 0,
    }


def _fetch_setlistfm(nombre: str, api_key: str) -> dict:
    import os as _os
    _os.environ.setdefault("SETLIST_FM_API_KEY", api_key)

    from src.data_collectors.setlistfm import buscar_mbid_por_nombre, obtener_setlists, parsear_setlist

    resultado_busqueda = buscar_mbid_por_nombre(nombre)
    if not resultado_busqueda:
        nombre_norm = _normalizar_nombre(nombre)
        if nombre_norm != nombre.lower().strip():
            resultado_busqueda = buscar_mbid_por_nombre(nombre_norm)
    if not resultado_busqueda:
        return {}

    mbid, _ = resultado_busqueda
    setlists = obtener_setlists(str(mbid), max_paginas=3)

    base = {"sl_tiene_datos": 1, "sl_num_conciertos": 0,
            "sl_avg_canciones": 0.0, "sl_pct_encore": 0.0,
            "sl_num_paises": 0, "sl_pct_espana": 0.0}

    if not setlists:
        return base

    conciertos = [parsear_setlist(s, str(mbid), nombre) for s in setlists]
    validos = [c for c in conciertos if c.get("fecha")]

    if not validos:
        return base

    n = len(validos)
    avg_canciones = sum(c.get("num_canciones") or 0 for c in validos) / n
    pct_encore = sum(1 for c in validos if c.get("tiene_encore")) / n
    paises = {c.get("pais") for c in validos if c.get("pais")}
    num_espana = sum(1 for c in validos if c.get("pais") == "Spain")

    return {
        "sl_tiene_datos":    1,
        "sl_num_conciertos": n,
        "sl_avg_canciones":  round(avg_canciones, 2),
        "sl_pct_encore":     round(pct_encore, 4),
        "sl_num_paises":     len(paises),
        "sl_pct_espana":     round(num_espana / n, 4),
    }


def _fetch_gtrends(nombre: str, timeout: float = 15.0) -> dict:
    """Solo Google Trends. Sin dependencias externas — se lanza en el pool principal."""
    result = {"trend_gtrends_interes_medio": 0.0}
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="es-ES", tz=60, timeout=(8, timeout))
        pt.build_payload([nombre], geo="ES", timeframe="today 12-m")
        df = pt.interest_over_time()
        if not df.empty and nombre in df.columns:
            result["trend_gtrends_interes_medio"] = round(float(df[nombre].mean()), 2)
    except Exception:
        pass
    return result


def _fetch_yt_reciente(channel_id: Optional[str], youtube_api_key: str) -> dict:
    """YouTube reciente. Requiere channel_id de YouTube — se ejecuta en fase 2."""
    result = {"trend_yt_vistas_recientes": 0.0, "trend_yt_videos_recientes": 0}
    if not channel_id or not youtube_api_key:
        return result
    try:
        os.environ.setdefault("YOUTUBE_API_KEY", youtube_api_key)
        from src.data_collectors.tendencias import get_youtube_reciente
        yt = get_youtube_reciente(channel_id)
        result["trend_yt_vistas_recientes"] = float(yt.get("yt_vistas_recientes") or 0)
        result["trend_yt_videos_recientes"] = int(yt.get("yt_videos_recientes") or 0)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def fetch_features_por_nombre(
    nombre: str,
    spotify_id: str,
    nombre_spotify: str,
    *,
    sp_client=None,
    lastfm_api_key: Optional[str] = None,
    youtube_api_key: Optional[str] = None,
    setlistfm_api_key: Optional[str] = None,
    tendencias_timeout: float = 15.0,
    on_progress=None,   # callback(plataforma, estado, datos) para actualizar UI
) -> ResultadoFetch:
    """
    Obtiene datos de todas las plataformas en paralelo para un artista confirmado.
    Nunca lanza excepción: los errores quedan en resultado.estado[plataforma].

    on_progress: callable(plataforma: str, estado: str, datos: dict) — se llama
    conforme va completando cada plataforma para actualizar el spinner en Streamlit.
    """
    lastfm_key = lastfm_api_key or os.getenv("LASTFM_API_KEY", "")
    yt_key = youtube_api_key or os.getenv("YOUTUBE_API_KEY", "")
    sl_key = setlistfm_api_key or os.getenv("SETLIST_FM_API_KEY", "")

    resultado = ResultadoFetch(
        nombre_spotify=nombre_spotify,
        spotify_id=spotify_id,
        estado={p: "skipped" for p in ("spotify", "lastfm", "youtube", "setlistfm", "tendencias")},
    )

    tareas = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        if sp_client:
            tareas["spotify"]   = executor.submit(_fetch_spotify, sp_client, spotify_id, nombre_spotify)
        if lastfm_key:
            tareas["lastfm"]    = executor.submit(_fetch_lastfm, nombre_spotify, lastfm_key)
        if yt_key:
            tareas["youtube"]   = executor.submit(_fetch_youtube, nombre_spotify, yt_key)
        if sl_key:
            tareas["setlistfm"] = executor.submit(_fetch_setlistfm, nombre_spotify, sl_key)
        if yt_key:
            tareas["gtrends"]   = executor.submit(_fetch_gtrends, nombre_spotify, tendencias_timeout)

        futures_map = {v: k for k, v in tareas.items()}
        try:
            for future in as_completed(futures_map, timeout=35):
                plataforma = futures_map[future]
                try:
                    datos = future.result()
                    if datos:
                        resultado.estado[plataforma] = "ok"
                        if plataforma == "gtrends":
                            resultado.trend.update(datos)
                        else:
                            setattr(resultado, _plataforma_attr(plataforma), datos)
                    else:
                        resultado.estado[plataforma] = "not_found"
                        resultado.advertencias.append(_aviso_not_found(plataforma))
                except Exception as e:
                    resultado.estado[plataforma] = "error"
                    resultado.advertencias.append(f"{plataforma}: error al conectar ({type(e).__name__})")

                if on_progress and plataforma != "gtrends":
                    on_progress(plataforma, resultado.estado[plataforma],
                                getattr(resultado, _plataforma_attr(plataforma), {}))
        except TimeoutError:
            resultado.advertencias.append("Algunas plataformas no respondieron a tiempo (>35 s).")

    # Fase 2: YouTube reciente (necesita channel_id resuelto por _fetch_youtube)
    if yt_key:
        channel_id = resultado.yt.get("channel_id")
        try:
            yt_rec = _fetch_yt_reciente(channel_id, yt_key)
            resultado.trend.update(yt_rec)
            resultado.estado["tendencias"] = "ok" if any(resultado.trend.values()) else "not_found"
        except Exception as e:
            resultado.estado["tendencias"] = "error"
            resultado.advertencias.append(f"tendencias: error ({type(e).__name__})")

        if on_progress:
            on_progress("tendencias", resultado.estado["tendencias"], resultado.trend)

    return resultado


def _plataforma_attr(plataforma: str) -> str:
    _MAP = {
        "spotify":    "sp",
        "lastfm":     "lfm",
        "youtube":    "yt",
        "setlistfm":  "sl",
        "tendencias": "trend",
    }
    return _MAP.get(plataforma, plataforma)


def _aviso_not_found(plataforma: str) -> str:
    avisos = {
        "spotify":   "Spotify: no se pudieron obtener los datos de discografía.",
        "lastfm":    "Last.fm: artista no encontrado. Si tiene perfil, introduce los datos manualmente.",
        "youtube":   "YouTube: canal no encontrado. Si tiene canal, introduce los datos manualmente.",
        "setlistfm": "setlist.fm: artista no encontrado. Si ha dado conciertos, introdúcelos manualmente.",
        "tendencias": "Tendencias: sin datos (rate limit o artista poco buscado).",
    }
    return avisos.get(plataforma, f"{plataforma}: no encontrado.")


# ---------------------------------------------------------------------------
# Transformación ResultadoFetch → kwargs para predecir()
# ---------------------------------------------------------------------------

def resultado_a_features(resultado: ResultadoFetch) -> dict:
    """
    Transforma un ResultadoFetch en el dict de kwargs para predecir().
    Los valores ausentes (plataforma no encontrada) quedan a 0.
    """
    sp  = resultado.sp
    lfm = resultado.lfm
    yt  = resultado.yt
    sl  = resultado.sl
    tr  = resultado.trend

    return {
        # Spotify
        "sp_num_albums":      int(sp.get("sp_num_albums", 0)),
        "sp_num_singles":     int(sp.get("sp_num_singles", 0)),
        "sp_anos_activo":     float(sp.get("sp_anos_activo", 1.0)),
        "sp_avg_duration_ms": float(sp.get("sp_avg_duration_ms", 210_000)),
        "sp_pct_explicit":    float(sp.get("sp_pct_explicit", 0.6)),
        "sp_pct_colabs":      float(sp.get("sp_pct_colabs", 0.3)),
        # Last.fm
        "lfm_oyentes":     int(lfm.get("lfm_oyentes", 0)),
        "lfm_scrobbles":   int(lfm.get("lfm_scrobbles", 0)),
        "lfm_num_generos": int(lfm.get("lfm_num_generos", 0)),
        # YouTube
        "yt_suscriptores":    int(yt.get("yt_suscriptores", 0)),
        "yt_vistas_totales":  int(yt.get("yt_vistas_totales", 0)),
        "yt_num_videos":      int(yt.get("yt_num_videos", 0)),
        # setlist.fm
        "sl_num_conciertos": int(sl.get("sl_num_conciertos", 0)),
        "sl_tiene_datos":    int(sl.get("sl_tiene_datos", 0)),
        "sl_avg_canciones":  float(sl.get("sl_avg_canciones", 0.0)),
        "sl_pct_encore":     float(sl.get("sl_pct_encore", 0.0)),
        "sl_num_paises":     int(sl.get("sl_num_paises", 0)),
        "sl_pct_espana":     float(sl.get("sl_pct_espana", 0.0)),
        # Tendencias
        "trend_gtrends_interes_medio": float(tr.get("trend_gtrends_interes_medio", 0.0)),
        "trend_yt_vistas_recientes":   float(tr.get("trend_yt_vistas_recientes", 0.0)),
        "trend_yt_videos_recientes":   int(tr.get("trend_yt_videos_recientes", 0)),
    }
