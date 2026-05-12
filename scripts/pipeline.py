"""
Orquestador del pipeline completo: recolección → features → entrenamiento.

Prerrequisito: config/artistas_registry.csv debe existir y estar actualizado.
  - Primera vez: python scripts/migrar_registry.py
  - Artistas nuevos: python scripts/00_resolver_identidades.py
    → revisar pendientes_revision.csv → completar registry manualmente
    → volver a ejecutar este script

Uso:
  python scripts/pipeline.py              # recolección + features + entrenamiento
  python scripts/pipeline.py --no-train   # solo recolección + features
  python scripts/pipeline.py --only-features  # solo rebuild de features (sin APIs)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: list[str], descripcion: str) -> bool:
    """Ejecuta un comando y devuelve True si tiene éxito."""
    print(f"\n{'='*60}")
    print(f"  {descripcion}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, "-m"] + cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  ERROR en '{descripcion}' (código {result.returncode})")
        print("  El pipeline se ha detenido. Corrige el error y vuelve a ejecutar.")
        return False
    return True


def run_script(script: str, descripcion: str) -> bool:
    """Ejecuta un script directamente (no como módulo)."""
    print(f"\n{'='*60}")
    print(f"  {descripcion}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  ERROR en '{descripcion}' (código {result.returncode})")
        print("  El pipeline se ha detenido. Corrige el error y vuelve a ejecutar.")
        return False
    return True


def verificar_registry():
    """Comprueba que el registry existe y advierte si hay artistas nuevos sin resolver."""
    sys.path.insert(0, str(ROOT))
    from config.registry import cargar as cargar_registry, REGISTRY_PATH

    try:
        df_reg = cargar_registry()
    except FileNotFoundError:
        print("ERROR: config/artistas_registry.csv no encontrado.")
        print("Ejecuta primero:")
        print("  python scripts/migrar_registry.py   (si ya tienes CSVs recolectados)")
        print("  python scripts/00_resolver_identidades.py  (artistas nuevos)")
        sys.exit(1)

    artistas_file = ROOT / "artistas.txt"
    with open(artistas_file, encoding="utf-8") as f:
        artistas_txt = {l.strip().lower() for l in f if l.strip()}

    en_registry = set(df_reg["nombre_canonico"].str.lower())
    sin_registry = artistas_txt - en_registry

    if sin_registry:
        print(f"\nAVISO: {len(sin_registry)} artistas en artistas.txt sin entrada en el registry:")
        for a in sorted(sin_registry):
            print(f"  - {a}")
        print("\nEjecuta python scripts/00_resolver_identidades.py antes del pipeline.")
        print("(O continúa si quieres procesar solo los artistas ya en el registry)\n")

    sin_resolver = df_reg[
        df_reg["spotify_id"].isna() | df_reg["lastfm_nombre"].isna() |
        df_reg["yt_channel_id"].isna()
    ]
    if not sin_resolver.empty:
        print(f"INFO: {len(sin_resolver)} artistas con alguna plataforma sin resolver en el registry.")
        print("Los datos de esas plataformas aparecerán como NaN en las features.\n")

    return df_reg


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de recolección y entrenamiento")
    parser.add_argument("--no-train",       action="store_true", help="Omitir entrenamiento del modelo")
    parser.add_argument("--only-features",  action="store_true", help="Solo rebuild de features (sin APIs)")
    parser.add_argument("--batch-size-yt",  type=int, default=0, help="Batch size para YouTube")
    args = parser.parse_args()

    print("Pipeline musical_predictor_PFM")
    print("="*60)

    verificar_registry()

    if not args.only_features:
        # ── Recolección de datos ──────────────────────────────────────────────
        pasos = [
            ("src.data_collectors.spotify_features", "Spotify: discografía y top tracks"),
            ("src.data_collectors.lastfmapi",        "Last.fm: oyentes y scrobbles"),
            ("src.data_collectors.setlistfm",        "setlist.fm: historial de conciertos"),
            ("src.data_collectors.tendencias",       "Tendencias: Google Trends + YouTube reciente"),
        ]

        for modulo, desc in pasos:
            if not run(modulo.split("."), desc):
                sys.exit(1)

        # YouTube con batch opcional
        yt_cmd = ["src.data_collectors.youtube"]
        if args.batch_size_yt > 0:
            yt_cmd = ["src.data_collectors.youtube", "--batch-size", str(args.batch_size_yt)]
        if not run(yt_cmd, f"YouTube: estadísticas de canal{' (batch=' + str(args.batch_size_yt) + ')' if args.batch_size_yt else ''}"):
            sys.exit(1)

    # ── Feature engineering ───────────────────────────────────────────────────
    if not run_script("scripts/generar_labels.py", "Generar labels de tier_sala"):
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Features: construir matriz de features")
    print(f"{'='*60}")
    sys.path.insert(0, str(ROOT))
    from src.features.build_features import build_artist_features, save_features
    df = build_artist_features()
    out = save_features(df)
    print(f"  Shape: {df.shape}")
    print(f"  Guardado en: {out}")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    if not args.no_train:
        if not run(["src.models.train"], "Entrenamiento del modelo XGBoost"):
            sys.exit(1)
        if not run_script("scripts/generar_shap.py", "Generar plots SHAP globales"):
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Pipeline completado correctamente.")
    if args.no_train:
        print("  El modelo NO se ha reentrenado (--no-train).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
