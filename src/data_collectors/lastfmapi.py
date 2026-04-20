"""
Obtiene datos de artistas desde Last.fm usando los nombres de artistas.txt.
Extrae: oyentes mensuales, scrobbles totales, géneros (tags) y biografía corta.
Los resultados se cruzan con spotify_ids.csv para añadir el artist_id de Spotify.
"""

import os
import time
import difflib
import requests
import pandas as pd
from dotenv import load_dotenv
import argparse
from pathlib import Path

load_dotenv()

API_KEY  = os.getenv('LASTFM_API_KEY')
BASE_URL = 'http://ws.audioscrobbler.com/2.0/'


class LastFMCollector:
    def __init__(self):
        if not API_KEY:
            raise ValueError("❌ LASTFM_API_KEY no encontrada en .env")
        print("✓ Cliente de Last.fm inicializado correctamente")

    def _get(self, params: dict) -> dict:
        params.update({'api_key': API_KEY, 'format': 'json'})
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def buscar_artista(self, nombre: str, umbral: float = 0.6) -> dict:
        try:
            # Buscar varios candidatos y elegir el más parecido
            busqueda = self._get({'method': 'artist.search', 'artist': nombre, 'limit': 5})
            candidatos = busqueda.get('results', {}).get('artistmatches', {}).get('artist', [])

            if not candidatos:
                print(f"⚠️  No se encontró: {nombre}")
                return None

            mejor = max(candidatos, key=lambda a: self._similitud(nombre, a['name']))
            score = self._similitud(nombre, mejor['name'])

            if score < umbral:
                print(f"⚠️  Match rechazado: '{nombre}' → '{mejor['name']}' (similitud {score:.0%} < {umbral:.0%})")
                return None

            # Obtener datos completos del candidato elegido
            data = self._get({'method': 'artist.getInfo', 'artist': mejor['name']})
            if 'error' in data:
                print(f"⚠️  No encontrado: {nombre} ({data.get('message')})")
                return None

            a = data['artist']
            tags = [t['name'] for t in a.get('tags', {}).get('tag', [])]
            stats = a.get('stats', {})

            datos = {
                'nombre_buscado': nombre,
                'nombre_lastfm':  a.get('name'),
                'oyentes':        int(stats.get('listeners', 0)),
                'scrobbles':      int(stats.get('playcount', 0)),
                'generos':        ', '.join(tags) if tags else 'Sin género',
                'lastfm_url':     a.get('url'),
                'mbid':           a.get('mbid', ''),
                'match_score':    round(score, 2),
            }

            print(f"✓ {datos['nombre_lastfm']} (similitud {score:.0%}): {datos['oyentes']:,} oyentes | {datos['generos']}")
            return datos

        except Exception as e:
            print(f"❌ Error al buscar {nombre}: {str(e)}")
            return None

    def procesar_lista(self, lista_artistas: list, delay: float = 0.5) -> pd.DataFrame:
        resultados = []
        no_encontrados = []

        print(f"\n🎵 Procesando {len(lista_artistas)} artistas...\n")

        for i, nombre in enumerate(lista_artistas, 1):
            print(f"[{i}/{len(lista_artistas)}] Buscando: {nombre}")

            datos = None
            for intento in range(3):
                try:
                    datos = self.buscar_artista(nombre.strip())
                    break
                except Exception as e:
                    espera = 10 * (intento + 1)
                    print(f"  ⚠️  Error de conexión (intento {intento+1}/3), reintentando en {espera}s: {e}")
                    time.sleep(espera)
            else:
                print(f"  ❌ Sin datos tras 3 intentos: {nombre}")

            if datos:
                resultados.append(datos)
            else:
                # Fila vacía para rellenar manualmente
                resultados.append({
                    'nombre_buscado': nombre,
                    'nombre_lastfm': None,
                    'oyentes': None,
                    'scrobbles': None,
                    'generos': None,
                    'lastfm_url': None,
                    'mbid': None,
                    'match_score': None,
                })
                no_encontrados.append(nombre)

            time.sleep(delay)

        df = pd.DataFrame(resultados)
        con_datos = len(df) - len(no_encontrados)
        print(f"\n✓ Procesados {len(df)} artistas ({con_datos} con datos, {len(no_encontrados)} para rellenar manualmente)")

        if no_encontrados:
            print(f"\n📝 Rellenar manualmente ({len(no_encontrados)}):")
            for a in no_encontrados:
                print(f"   - {a}")

        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artists', type=str, default='artistas.txt')
    parser.add_argument('--spotify-ids', type=str, default='data/raw/spotify_ids.csv')
    parser.add_argument('--output', type=str, default='data/raw/lastfm_artistas.csv')
    args = parser.parse_args()

    if not os.path.exists(args.artists):
        print(f"❌ No se encuentra el archivo: {args.artists}")
        return

    with open(args.artists, 'r', encoding='utf-8') as f:
        artistas = [line.strip() for line in f if line.strip()]

    print(f"📋 {len(artistas)} artistas encontrados en {args.artists}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Cargar CSV existente y omitir artistas ya procesados
    df_existente = pd.DataFrame()
    ya_procesados = set()
    if os.path.exists(args.output):
        df_existente = pd.read_csv(args.output)
        ya_procesados = set(df_existente['nombre_buscado'].str.lower())
        print(f"📂 {len(df_existente)} artistas ya en {args.output} — se omitirán")

    nuevos = [a for a in artistas if a.lower() not in ya_procesados]
    if not nuevos:
        print("✓ No hay artistas nuevos que procesar")
        return

    print(f"🆕 {len(nuevos)} artistas nuevos a procesar")

    collector = LastFMCollector()
    df_nuevos = collector.procesar_lista(nuevos)

    # Cruzar con spotify_ids.csv para añadir artist_id a los nuevos
    if os.path.exists(args.spotify_ids):
        df_spotify = pd.read_csv(args.spotify_ids)[['nombre_buscado', 'artist_id', 'spotify_uri']]
        df_nuevos = df_nuevos.merge(df_spotify, on='nombre_buscado', how='left')
        print(f"🔗 Cruzado con {args.spotify_ids}")

    df = pd.concat([df_existente, df_nuevos], ignore_index=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n💾 Datos guardados en: {args.output}")
    print(f"📊 Total: {len(df)} artistas ({len(df_existente)} previos + {len(df_nuevos)} nuevos)")

    if not df.empty:
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Total artistas: {len(df)}")
        print(f"   Promedio oyentes: {df['oyentes'].mean():,.0f}")
        print(f"   Promedio scrobbles: {df['scrobbles'].mean():,.0f}")
        print(f"   Artista más escuchado: {df.loc[df['oyentes'].idxmax(), 'nombre_lastfm']} ({df['oyentes'].max():,} oyentes)")


if __name__ == '__main__':
    main()
