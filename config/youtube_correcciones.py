"""
youtube_correcciones.py
========================
Audit trail de correcciones manuales sobre youtube_artistas.csv.

Problema que resuelve:
    La búsqueda automática de YouTubeCollector.buscar_canal() no siempre
    encuentra el canal correcto (nombres ambiguos, canales compartidos,
    Topic channels, etc.). Cuando esto ocurre, el canal se corrige a mano
    editando el CSV directamente, sin dejar rastro de la decisión.

    Este módulo documenta esas correcciones y permite que el colector
    las aplique automáticamente en futuras re-ejecuciones, eliminando la
    necesidad de editar el CSV a mano y garantizando reproducibilidad.

Uso:
    from config.youtube_correcciones import CORRECCIONES, SIN_CANAL

    # El colector consulta este dict antes de hacer la llamada a la API.
    # Si el artista tiene entrada aquí, se usa directamente sin búsqueda.

Mantenimiento:
    Cada vez que corrijas un canal manualmente, añade una entrada en
    CORRECCIONES con el channel_id correcto y el motivo del fallo.
    Para artistas sin canal, añade una entrada en SIN_CANAL.
"""

# ---------------------------------------------------------------------------
# Artistas cuyo canal fue corregido manualmente tras la búsqueda automática
# ---------------------------------------------------------------------------
# Formato:
#   "nombre_buscado": {
#       "channel_id":   str   — ID del canal correcto (empieza por UC...)
#       "nombre_canal": str   — Nombre visible del canal en YouTube
#       "match_score":  float — Score de confianza (1.0 = corrección manual exacta)
#       "motivo":       str   — Por qué falló la búsqueda automatizada
#   }
#
# Para obtener el channel_id de un canal:
#   1. Abre el canal en YouTube
#   2. URL → https://www.youtube.com/@nombrecanal/about
#   3. Botón derecho → Ver código fuente → buscar "channelId"
#
# IMPORTANTE: Rellena este dict con los ~10 artistas que corregiste manualmente.
# ---------------------------------------------------------------------------

CORRECCIONES: dict[str, dict] = {
    # Ejemplo de cómo rellenar una entrada:
    # "Nombre Artista": {
    #     "channel_id":   "UCxxxxxxxxxxxxxxxxxxxxxxxxx",
    #     "nombre_canal": "Nombre Canal YouTube",
    #     "match_score":  1.0,
    #     "motivo": "La búsqueda devolvía el canal de un artista con nombre similar",
    # },
}


# ---------------------------------------------------------------------------
# Artistas confirmados sin canal de YouTube
# ---------------------------------------------------------------------------
# Sus features de YouTube quedan como NaN en artist_features.csv y se imputan
# a 0 (presencia digital nula) en preprocess.py.

SIN_CANAL: dict[str, str] = {
    "Xico Palma": "No tiene canal de YouTube verificado",
    "GREKKY":     "No tiene canal de YouTube verificado",
}
