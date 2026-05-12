"""
SpotifyCollector — búsqueda de artistas por nombre con validación de similitud.
Usado por scripts/00_resolver_identidades.py para poblar el registry.
La recolección de datos de Spotify usa spotify_features.py con IDs del registry.
"""

import os
import time
import difflib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()


class SpotifyCollector:
    def __init__(self):
        client_id     = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET no encontrados en .env")
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        ))

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def buscar_artista(self, nombre: str, umbral: float = 0.6) -> dict | None:
        """
        Busca un artista en Spotify por nombre.
        Devuelve dict con artist_id, nombre_spotify, etc. o None si el match no supera el umbral.
        """
        try:
            resultados = self.sp.search(q=f"artist:{nombre}", type="artist", limit=5)
            items = resultados["artists"]["items"]
            if not items:
                return None
            mejor = max(items, key=lambda a: self._similitud(nombre, a["name"]))
            score = self._similitud(nombre, mejor["name"])
            if score < umbral:
                print(f"  Match rechazado: '{nombre}' → '{mejor['name']}' ({score:.0%})")
                return None
            return {
                "nombre_buscado":  nombre,
                "nombre_spotify":  mejor["name"],
                "artist_id":       mejor["id"],
                "spotify_uri":     mejor["uri"],
                "spotify_url":     mejor.get("external_urls", {}).get("spotify"),
                "match_score":     round(score, 2),
            }
        except Exception as e:
            print(f"  Error al buscar '{nombre}': {e}")
            return None
        finally:
            time.sleep(0.3)
