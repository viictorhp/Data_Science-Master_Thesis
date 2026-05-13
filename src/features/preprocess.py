"""
preprocess.py
=============
Preprocesa artist_features.csv y devuelve X, y listos para entrenar.

VARIABLE OBJETIVO (y)
---------------------
  y = nivel  →  {bajo: 1, medio: 2, alto: 3}

  Clasificación multiclase ordinal: predecimos el tier de la sala más grande
  que un artista puede llenar en España. Es la columna 'target' del CSV,
  codificación numérica de 'nivel' (bajo/medio/alto).

VARIABLES PREDICTORAS (X)
--------------------------
  Todas las features del CSV excepto las columnas de metadata/target y las
  features eliminadas en el EDA por baja señal (ver FEATURES_ELIMINAR).

  Para modelos de árbol  → 32 features, sin escalar
  Para regresión lineal  → 24 features (sin versiones raw cuando hay log), RobustScaler

Uso:
  from src.features.preprocess import cargar_datos

  X_tree,   y, nombres = cargar_datos(modo='arbol')
  X_linear, y, nombres = cargar_datos(modo='lineal')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"
CSV_PATH  = PROCESSED / "artist_features.csv"

# ---------------------------------------------------------------------------
# Columnas que NO son features
# ---------------------------------------------------------------------------
# nombre_buscado: identificador del artista, no contiene información predictiva
# nivel, tier_sala, target: son el target o codificaciones del mismo
COLUMNAS_META = ["nombre_buscado", "nivel", "tier_sala", "target"]

# ---------------------------------------------------------------------------
# Features eliminadas por baja señal (justificación del EDA)
# ---------------------------------------------------------------------------
# sp_num_eps               → H≈0, r≈0. Sin varianza útil: casi todos los artistas
#                            tienen 0 EPs. No aporta ninguna señal discriminativa.
#
# sp_num_total_releases    → H≈2, r≈-0.04. Redundante: es suma de albums+singles+eps.
#                            La información ya está en esas columnas por separado.
#
# sp_dias_desde_ultimo_release → H≈2, r≈+0.10. No supera el umbral de significación.
#                            El EDA muestra que la actividad reciente no discrimina
#                            los niveles — tanto artistas bajo como alto pueden llevar
#                            meses sin lanzar.
#
# yt_edad_canal_anos       → H≈1, r≈+0.10. No significativa. La antigüedad del canal
#                            no discrimina niveles: hay artistas alto con canales
#                            recientes y artistas bajo con canales muy antiguos.
FEATURES_ELIMINAR = [
    "sp_num_eps",
    "sp_num_total_releases",
    "sp_dias_desde_ultimo_release",
    "yt_edad_canal_anos",
    # EDA 157 artistas: varianza casi nula (>95% de artistas tienen 5 vídeos recientes)
    "trend_yt_videos_recientes",
]

# ---------------------------------------------------------------------------
# Features raw eliminadas en modo lineal (solo se usan las versiones log)
# ---------------------------------------------------------------------------
# Para regresión logística, las versiones raw y log de la misma variable
# tienen correlación ≈1.0 (multicolinealidad perfecta). Además, las versiones
# raw tienen distribuciones muy asimétricas con outliers extremos (Quevedo,
# Bad Gyal) que distorsionan los coeficientes aunque usemos RobustScaler.
# Se conserva solo la versión log, que tiene distribución más normal.
FEATURES_RAW_SOLO_ARBOL = [
    "lfm_oyentes",                   # reemplazada por lfm_oyentes_log
    "lfm_scrobbles",                 # reemplazada por lfm_scrobbles_log
    "yt_suscriptores",               # reemplazada por yt_suscriptores_log
    "yt_vistas_totales",             # reemplazada por yt_vistas_log
    # sp_pct_colabs: H≈10 pero r≈-0.02, sin señal real sobre el target.
    # En modo lineal se elimina para no añadir ruido al modelo.
    "sp_pct_colabs",
    # Tendencias: distribuciones muy sesgadas (rango 0–3M+ vistas), se usan versiones log en modo lineal
    "trend_yt_vistas_recientes",          # reemplazada por trend_yt_vistas_recientes_log
    "trend_gtrends_interes_medio",        # reemplazada por trend_gtrends_log
    # EDA 157 artistas: r=0.996 con trend_yt_vistas_recientes_log → multicolinealidad casi perfecta
    "trend_yt_vistas_por_video_reciente",
]


# ---------------------------------------------------------------------------
# Imputación de NaN
# ---------------------------------------------------------------------------

def _imputar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores nulos según la naturaleza de cada feature.
    Justificación por grupo:

    Conteos de setlist.fm (sl_num_conciertos, sl_num_paises) → 0
        Un artista sin registros en setlist.fm tiene efectivamente 0 conciertos
        documentados. Imputar con la media introduciría información falsa;
        imputar con 0 es conservador y semánticamente correcto.

    Ratios de setlist.fm (sl_avg_canciones, sl_pct_encore, sl_pct_espana) → mediana
        Son ratios calculados sobre conciertos: no tiene sentido asignar 0
        (implicaría 0 canciones por show o 0% de encores, que es distinto a
        "no hay datos"). La mediana del conjunto es la estimación más neutral.

    YouTube vistas/videos y derivados → mediana por nivel
        Los artistas con NaN en estas columnas son aquellos cuyos canales fueron
        corregidos manualmente y no tenemos datos de vistas. Usar la mediana
        global mezclaría artistas de distinto nivel; usar la mediana dentro del
        mismo nivel es más precisa como estimación del valor desconocido.

    Resto (< 4% NaN: sp_anos_activo, sp_releases_por_ano, yt_suscriptores,
    lfm_oyentes, lfm_scrobbles) → 0
        Artistas sin lanzamientos en Spotify, sin canal YouTube o no encontrados
        en Last.fm. El 0 es el valor real en todos estos casos: 0 años activo,
        0 suscriptores, 0 oyentes.
    """
    df = df.copy()

    # sl_tiene_datos: debe capturarse ANTES de rellenar NaN en sl_num_conciertos
    # 1 = artista con perfil en setlist.fm (tenía datos reales), 0 = sin datos (NaN)
    # Validado con scripts/validar_setlistfm.py: el NaN está correlacionado con el tier
    # (79% de artistas "bajo" sin datos vs 5% de "alto"), por lo que hacer esta señal
    # explícita es más honesto que dejar que el modelo la aprenda implícitamente del 0 imputado.
    if "sl_num_conciertos" in df.columns:
        df["sl_tiene_datos"] = df["sl_num_conciertos"].notna().astype(int)

    # Conteos de directo → 0
    for col in ["sl_num_conciertos", "sl_num_paises"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Ratios de directo → mediana global
    for col in ["sl_avg_canciones", "sl_pct_encore", "sl_pct_espana"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # YouTube vistas/videos → mediana por nivel
    yt_cols_vistas = [
        "yt_vistas_totales", "yt_vistas_log",
        "yt_num_videos", "yt_vistas_por_video", "yt_vistas_por_suscriptor",
    ]
    for col in yt_cols_vistas:
        if col in df.columns:
            df[col] = df.groupby("nivel")[col].transform(
                lambda s: s.fillna(s.median())
            )

    # Google Trends None = fallo de rate-limit, valor desconocido → mediana por nivel
    for col in ["trend_gtrends_interes_medio", "trend_gtrends_log"]:
        if col in df.columns:
            df[col] = df.groupby("nivel")[col].transform(
                lambda s: s.fillna(s.median())
            )

    # YouTube reciente None = artista sin canal → 0 vistas es semánticamente correcto
    for col in [
        "trend_yt_vistas_recientes", "trend_yt_vistas_recientes_log",
        "trend_yt_videos_recientes", "trend_yt_vistas_por_video_reciente",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Resto → 0
    df = df.fillna(0)

    return df


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def cargar_datos(
    modo: str = "arbol",
    csv_path: Path = CSV_PATH,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Carga, imputa y prepara X e y para el entrenamiento.

    Parámetros
    ----------
    modo : 'arbol' | 'lineal'
        'arbol'  → 32 features, sin escalar. Para Random Forest y XGBoost.
                   Los árboles son invariantes al escalado y toleran features
                   correlacionadas (seleccionan la mejor en cada split).
        'lineal' → 24 features (sin versiones raw), RobustScaler aplicado.
                   Para Regresión Logística y SVM.
                   RobustScaler usa mediana e IQR en lugar de media y desviación
                   típica, por lo que es robusto ante los outliers extremos
                   identificados en el EDA (Quevedo, Bad Gyal, Natos y Waor).

    Retorna
    -------
    X        : DataFrame con las features seleccionadas e imputadas
    y        : Series con el target (1=bajo, 2=medio, 3=alto)
    features : lista de nombres de columnas de X
    """
    if modo not in ("arbol", "lineal"):
        raise ValueError(f"modo debe ser 'arbol' o 'lineal', recibido: '{modo}'")

    df_raw = pd.read_csv(csv_path)

    # --- Target ---
    # Usamos 'target' (1/2/3) como variable numérica ordinal.
    # 'nivel' y 'tier_sala' son la misma información en otro formato — no se usan.
    y = df_raw["target"].astype(int)

    # --- Imputar antes de seleccionar features ---
    # La imputación de vistas usa 'nivel' para calcular medianas por grupo,
    # por eso imputamos sobre el DataFrame completo antes de separar X e y.
    df = _imputar(df_raw)

    # --- Seleccionar features ---
    excluir = set(COLUMNAS_META) | set(FEATURES_ELIMINAR)
    if modo == "lineal":
        excluir |= set(FEATURES_RAW_SOLO_ARBOL)

    features = [c for c in df.columns if c not in excluir]
    X = df[features].copy()

    # --- Escalar (solo modo lineal) ---
    if modo == "lineal":
        scaler = RobustScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=features,
            index=X.index,
        )

    return X, y, features


# ---------------------------------------------------------------------------
# Entry point — muestra resumen del resultado
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for modo in ("arbol", "lineal"):
        X, y, features = cargar_datos(modo=modo)
        print(f"\n{'='*50}")
        print(f"Modo: {modo}")
        print(f"X shape: {X.shape}  |  y shape: {y.shape}")
        print(f"NaN en X: {X.isna().sum().sum()}")
        print(f"Distribución del target:\n{y.value_counts().sort_index()}")
        print(f"\nFeatures ({len(features)}):")
        for f in features:
            print(f"  {f}")
