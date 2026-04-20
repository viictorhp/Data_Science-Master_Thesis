"""
Obtiene el Spotify ID (artist_id) y URI de cada artista en artistas.txt.
Estos identificadores se usan como referencia universal para cruzar con otras fuentes.
"""

import os
import time
import difflib
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import argparse

load_dotenv()


class SpotifyCollector:
    def __init__(self):
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

        if not client_id or not client_secret:
            raise ValueError("❌ Credenciales de Spotify no encontradas en .env")

        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
        print("✓ Cliente de Spotify inicializado correctamente")

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def buscar_artista(self, nombre_artista: str, umbral: float = 0.6) -> dict:
        try:
            # Buscar varios candidatos y quedarse con el más parecido al nombre buscado
            resultados = self.sp.search(q=f'artist:{nombre_artista}', type='artist', limit=5)
            items = resultados['artists']['items']

            if not items:
                print(f"⚠️  No se encontró: {nombre_artista}")
                return None

            # Elegir el candidato con mayor similitud de nombre
            mejor = max(items, key=lambda a: self._similitud(nombre_artista, a['name']))
            score = self._similitud(nombre_artista, mejor['name'])

            if score < umbral:
                print(f"⚠️  Match rechazado: '{nombre_artista}' → '{mejor['name']}' (similitud {score:.0%} < {umbral:.0%})")
                return None

            datos = {
                'nombre_buscado': nombre_artista,
                'nombre_spotify': mejor.get('name'),
                'artist_id': mejor.get('id'),
                'spotify_uri': mejor.get('uri'),
                'spotify_url': mejor.get('external_urls', {}).get('spotify'),
                'match_score': round(score, 2),
            }
            print(f"✓ {datos['nombre_buscado']} → {datos['nombre_spotify']} (similitud {score:.0%})")
            return datos

        except Exception as e:
            print(f"❌ Error al buscar {nombre_artista}: {str(e)}")
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
                    'nombre_spotify': None,
                    'artist_id': None,
                    'spotify_uri': None,
                    'spotify_url': None,
                    'match_score': None,
                })
                no_encontrados.append(nombre)

            time.sleep(delay)

        df = pd.DataFrame(resultados)
        print(f"\n✓ Procesados {len(df)} artistas ({len(df) - len(no_encontrados)} con datos, {len(no_encontrados)} para rellenar manualmente)")

        if no_encontrados:
            print(f"\n📝 Rellenar manualmente ({len(no_encontrados)}):")
            for a in no_encontrados:
                print(f"   - {a}")

        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artists', type=str, default='artistas.txt')
    parser.add_argument('--output', type=str, default='data/raw/spotify_ids.csv')
    args = parser.parse_args()

    if not os.path.exists(args.artists):
        print(f"❌ No se encuentra el archivo: {args.artists}")
        return

    with open(args.artists, 'r', encoding='utf-8') as f:
        artistas = [line.strip() for line in f if line.strip()]

    print(f"📋 {len(artistas)} artistas encontrados en {args.artists}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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

    collector = SpotifyCollector()
    df_nuevos = collector.procesar_lista(nuevos)

    df = pd.concat([df_existente, df_nuevos], ignore_index=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n💾 IDs guardados en: {args.output}")
    print(f"📊 Total: {len(df)} artistas ({len(df_existente)} previos + {len(df_nuevos)} nuevos)")


if __name__ == '__main__':
    main()
