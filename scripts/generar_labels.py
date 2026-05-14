"""
Genera artistas_labels.csv con una columna tier_sala por artista.

Lógica de inferencia (en orden de prioridad):
  1. Mapeo directo de venue_nombre a tier usando keywords de venues conocidos
  2. Para cada artista, se toma el tier del venue MAS GRANDE en que ha tocado
  3. Si no hay historial en setlist.fm, tier_inferido = NaN (rellenar manualmente)

Tiers:
  0 = sin capacidad de concierto propio
  1 = sala pequeña  (<200)
  2 = sala mediana  (200-600)
  3 = sala grande   (600-2000)
  4 = macrosala / festival

Uso:
  python scripts/generar_labels.py
  python scripts/generar_labels.py --setlists data/raw/setlistfm_conciertos.csv
"""

import argparse
import re
import pandas as pd
from pathlib import Path

# ── Diccionario de venues conocidos → tier ────────────────────────────────────
# Keywords en minúsculas; se busca si aparecen en el nombre del venue
# Ordenados de mayor a menor tier para que el primero que coincida gane
VENUE_TIER = [
    # Tier 4 — macrosala / festival (>2000)
    (4, [
        "wizink", "palau sant jordi", "palacio de los deportes", "palacio de deportes",
        "estadio", "stadium", "festival", "primavera sound", "mad cool", "sonorama",
        "benicassim", "fib", "arenal sound", "viña rock", "Download",
        "pabellón", "pabellon", "arena", "olimpic", "olympic",
        "complejo", "hipódromo", "hipodromo", "ifema",
    ]),
    # Tier 3 — sala grande (600-2000)
    (3, [
        "razzmatazz", "razz", "apolo", "la riviera", "riviera",
        "mon live", "mon-live", "la mar de músicas",
        "industrial copera", "copera",
        "auditorio garcia lorca", "garcia lorca",
        "auditorio nacional", "auditorio municipal",
        "palacio de congresos",
        "sala caracol", "caracol",
        "la salamandra", "salamandra",
        "sala bóveda", "boveda",
        "sala we", " we ",
        "joy eslava", "joy",
        "theater", "theatre", "teatro",
        "concert hall", "konzerthaus",
    ]),
    # Tier 2 — sala mediana (200-600)
    (2, [
        "planta baja", "sala el tren", "el tren",
        "sala aliatar", "aliatar",
        "auditorio manuel de falla", "manuel de falla",
        "sala el sol", "el sol",
        "cats", "freedom",
        "café berlín", "cafe berlin",
        "costello", "moby dick",
        "the clinic", "clinic",
        "sala 0", "sala zero",
        "jerusalem club",
        "fever blue",
        "sala x",
    ]),
    # Tier 1 — sala pequeña (<200)
    (1, [
        "sala vibora", "vibora",
        "plató", "plato",
        "bar ", "pub ", "cafe ", "café ",
        "club ", "local ",
    ]),
]


def inferir_tier_venue(venue_nombre: str) -> int | None:
    """Devuelve el tier del venue o None si no hay match."""
    if not isinstance(venue_nombre, str) or not venue_nombre.strip():
        return None
    nombre = venue_nombre.lower()
    for tier, keywords in VENUE_TIER:
        for kw in keywords:
            if kw in nombre:
                return tier
    return None


def tier_desde_historial(df_setlists: pd.DataFrame, nombre: str) -> tuple:
    """
    Dado el historial de setlists de un artista, devuelve
    (tier_inferido, venue_mayor) basándose en el venue con mayor tier.
    """
    filas = df_setlists[df_setlists["nombre"].str.lower() == nombre.lower()]
    if filas.empty:
        return None, None

    mejor_tier  = None
    mejor_venue = None

    for venue in filas["venue_nombre"].dropna().unique():
        t = inferir_tier_venue(venue)
        if t is not None and (mejor_tier is None or t > mejor_tier):
            mejor_tier  = t
            mejor_venue = venue

    return mejor_tier, mejor_venue


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists",  default="artistas.txt")
    parser.add_argument("--setlists", default="data/raw/setlistfm_conciertos.csv")
    parser.add_argument("--output",   default="config/artistas_labels.csv")
    args = parser.parse_args()

    # ── Cargar artistas ───────────────────────────────────────────────────────
    with open(args.artists, encoding="utf-8") as f:
        artistas = [l.strip() for l in f if l.strip()]
    print(f"📋 {len(artistas)} artistas en {args.artists}")

    # ── Cargar setlists ───────────────────────────────────────────────────────
    df_setlists = pd.DataFrame()
    if Path(args.setlists).exists():
        df_setlists = pd.read_csv(args.setlists)
        print(f"   {len(df_setlists)} filas en setlistfm_conciertos.csv")
    else:
        print(f"⚠️  No se encuentra {args.setlists}, tier_inferido será NaN para todos")

    # ── Inferir tier por artista ──────────────────────────────────────────────
    filas = []
    sin_historial = []
    sin_match     = []

    for nombre in artistas:
        tier_inf, venue_max = tier_desde_historial(df_setlists, nombre)

        if tier_inf is None and not df_setlists.empty:
            # Tiene filas en setlists pero ningún venue matcheado
            filas_art = df_setlists[df_setlists["nombre"].str.lower() == nombre.lower()]
            if filas_art.empty:
                sin_historial.append(nombre)
            else:
                sin_match.append(nombre)

        filas.append({
            "nombre_buscado": nombre,
            "tier_inferido":  tier_inf,
            "venue_mayor":    venue_max,
            "tier_sala":      tier_inf,   # columna definitiva — editar manualmente
            "notas":          "",
        })

    df = pd.DataFrame(filas)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    con_tier = df["tier_inferido"].notna().sum()
    print(f"\n📊 Resultados de inferencia:")
    print(f"   Con tier inferido:      {con_tier}")
    print(f"   Sin historial (setlists): {len(sin_historial)}")
    print(f"   Con historial sin match:  {len(sin_match)}")
    print(f"\n   Distribución de tiers inferidos:")
    for t in sorted(df["tier_inferido"].dropna().unique()):
        n = (df["tier_inferido"] == t).sum()
        label = {1: "pequeña <200", 2: "mediana 200-600", 3: "grande 600-2000", 4: "macrosala/festival"}
        print(f"     Tier {int(t)} ({label.get(int(t), '?')}): {n} artistas")

    if sin_historial:
        print(f"\n📝 Sin historial en setlist.fm ({len(sin_historial)}) — rellenar manualmente:")
        for a in sin_historial:
            print(f"   - {a}")

    if sin_match:
        print(f"\n🔍 Con historial pero venue no reconocido ({len(sin_match)}) — revisar:")
        for a in sin_match:
            filas_art = df_setlists[df_setlists["nombre"].str.lower() == a.lower()]
            venues = filas_art["venue_nombre"].dropna().unique()
            venues_str = ", ".join(str(v) for v in venues[:3])
            print(f"   - {a}: {venues_str}")

    # ── Preservar ediciones manuales de tier_sala ─────────────────────────────
    if Path(args.output).exists():
        df_prev = pd.read_csv(args.output)[['nombre_buscado', 'tier_sala', 'notas']]
        df = df.merge(df_prev, on='nombre_buscado', how='left', suffixes=('', '_prev'))
        existentes = df['tier_sala_prev'].notna()
        df.loc[existentes, 'tier_sala'] = df.loc[existentes, 'tier_sala_prev']
        df.loc[existentes, 'notas']     = df.loc[existentes, 'notas_prev'].fillna('')
        df = df.drop(columns=['tier_sala_prev', 'notas_prev'])
        nuevos_count = (~existentes).sum()
        print(f"   🔒 {existentes.sum()} artistas con tier_sala preservado")
        if nuevos_count:
            print(f"   🆕 {nuevos_count} artistas nuevos añadidos")

    # ── Guardar ───────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n💾 {args.output} generado ({len(df)} artistas)")
    print(f"   Rellena manualmente la columna 'tier_sala' para los que tienen NaN")


if __name__ == "__main__":
    main()
