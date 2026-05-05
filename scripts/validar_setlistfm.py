"""
validar_setlistfm.py
====================
Valida si la imputación a 0 de sl_num_conciertos (41% NaN) introduce
un artefacto en el modelo o refleja información real.

Pregunta central:
    ¿El NaN en setlist.fm está distribuido aleatoriamente entre tiers,
    o hay un patrón donde los artistas de nivel "bajo" tienen más NaN?
    Si hay patrón, el 0 imputado actúa como señal proxy del nivel bajo
    (artefacto). Si el NaN es aleatorio, la imputación es legítima.

Análisis que realiza:
    1. Tasa de NaN por tier — test chi-cuadrado de independencia
    2. Distribución de sl_num_conciertos entre artistas CON datos
    3. Comparativa CV5 con y sin las features de setlist.fm
    4. Recomendación final

Requiere:
    data/processed/artist_features.csv

Uso:
    python -m scripts.validar_setlistfm
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.preprocess import cargar_datos, CSV_PATH

SEED  = 42
ORDEN = ["bajo", "medio", "alto"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separador(titulo: str = "") -> None:
    linea = "=" * 60
    print(f"\n{linea}")
    if titulo:
        print(f"  {titulo}")
        print(linea)


def _cv5(model, X, y) -> dict:
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"],
                         return_train_score=False)
    return {
        "acc":  (res["test_accuracy"].mean(),  res["test_accuracy"].std()),
        "f1":   (res["test_f1_macro"].mean(),  res["test_f1_macro"].std()),
    }


def _modelo():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Análisis 1: ¿El NaN está correlacionado con el tier?
# ---------------------------------------------------------------------------

def analisis_nan_vs_tier(df_raw: pd.DataFrame) -> None:
    _separador("1. Tasa de NaN en sl_num_conciertos por tier")

    df_raw = df_raw.copy()
    df_raw["sin_datos"] = df_raw["sl_num_conciertos"].isna().astype(int)

    tabla = (
        df_raw.groupby("nivel")["sin_datos"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "sin_datos_sl", "count": "total"})
        .reindex(ORDEN)
    )
    tabla["pct_sin_datos"] = (tabla["sin_datos_sl"] / tabla["total"] * 100).round(1)
    print(tabla.to_string())

    # Test chi-cuadrado: ¿NaN es independiente del tier?
    contingencia = pd.crosstab(df_raw["nivel"], df_raw["sin_datos"])
    chi2, p, dof, _ = stats.chi2_contingency(contingencia)
    print(f"\nTest chi-cuadrado: χ²={chi2:.3f} · df={dof} · p={p:.4f}")

    if p < 0.05:
        print("\n⚠️  RESULTADO: El NaN NO es aleatorio (p < 0.05).")
        print("   Los artistas de ciertos tiers tienen más probabilidad de no")
        print("   aparecer en setlist.fm. El 0 imputado actúa como señal proxy")
        print("   del nivel → puede ser un ARTEFACTO parcial.")
    else:
        print("\n✅ RESULTADO: El NaN parece aleatorio entre tiers (p >= 0.05).")
        print("   La imputación a 0 no introduce un sesgo sistemático.")


# ---------------------------------------------------------------------------
# Análisis 2: Distribución de conciertos entre artistas CON datos
# ---------------------------------------------------------------------------

def analisis_distribucion_con_datos(df_raw: pd.DataFrame) -> None:
    _separador("2. Distribución de sl_num_conciertos (solo artistas CON datos)")

    df_con = df_raw.dropna(subset=["sl_num_conciertos"])
    total_con = len(df_con)
    total     = len(df_raw)
    print(f"  Artistas con datos: {total_con}/{total} ({total_con/total*100:.1f}%)")
    print(f"  Artistas sin datos: {total - total_con}/{total} ({(total-total_con)/total*100:.1f}%)\n")

    stats_tier = (
        df_con.groupby("nivel")["sl_num_conciertos"]
        .agg(["count", "median", "mean", "max"])
        .reindex(ORDEN)
        .round(1)
    )
    stats_tier.columns = ["n_con_datos", "mediana", "media", "max"]
    print("  Estadísticas por tier (artistas CON datos):")
    print(stats_tier.to_string())

    # ¿Cuántos artistas con 0 conciertos reales existen (sl=0, no NaN)?
    ceros_reales = (df_con["sl_num_conciertos"] == 0).sum()
    print(f"\n  Artistas con sl_num_conciertos = 0 (real, no imputado): {ceros_reales}")


# ---------------------------------------------------------------------------
# Análisis 3: Impacto en el modelo — con y sin features de setlist.fm
# ---------------------------------------------------------------------------

def analisis_impacto_modelo() -> None:
    _separador("3. Impacto en el modelo — CV5 con y sin features setlist.fm")

    print("  Cargando datos (modo árbol)...")
    X, y, features = cargar_datos(modo="arbol")
    y_xgb = y - 1  # 0-indexed

    # --- Modelo completo ---
    print("  Evaluando modelo COMPLETO (31 features)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_completo = _cv5(_modelo(), X, y_xgb)

    # --- Modelo sin features de setlist.fm ---
    sl_features = [f for f in features if f.startswith("sl_")]
    X_sin_sl    = X.drop(columns=sl_features)
    print(f"  Features eliminadas ({len(sl_features)}): {sl_features}")
    print(f"  Evaluando modelo SIN setlist.fm ({X_sin_sl.shape[1]} features)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_sin_sl = _cv5(_modelo(), X_sin_sl, y_xgb)

    print("\n  Resultados comparativos:")
    print(f"  {'Configuración':<35} {'Accuracy':>10} {'F1 macro':>10}")
    print(f"  {'-'*55}")
    acc_c, f1_c = res_completo["acc"], res_completo["f1"]
    acc_s, f1_s = res_sin_sl["acc"],   res_sin_sl["f1"]
    print(f"  {'Completo (31 features)':<35} {acc_c[0]:.3f}±{acc_c[1]:.3f}  {f1_c[0]:.3f}±{f1_c[1]:.3f}")
    print(f"  {'Sin setlist.fm (26 features)':<35} {acc_s[0]:.3f}±{acc_s[1]:.3f}  {f1_s[0]:.3f}±{f1_s[1]:.3f}")

    delta_acc = acc_c[0] - acc_s[0]
    delta_f1  = f1_c[0]  - f1_s[0]
    print(f"\n  Δ Accuracy: {delta_acc:+.3f}  |  Δ F1 macro: {delta_f1:+.3f}")

    print("\n  Interpretación:")
    if abs(delta_f1) < 0.02:
        print("  ✅ Las features de setlist.fm aportan <2 puntos de F1.")
        print("     Su eliminación no degrada el modelo significativamente.")
        print("     El modelo funciona bien sin ellas → el artefacto no es crítico.")
    elif delta_f1 > 0.05:
        print("  ⚠️  Las features de setlist.fm aportan >5 puntos de F1.")
        print("     Tienen señal real. El riesgo de artefacto es bajo:")
        print("     incluso si el 41% imputado tiene sesgo, la señal en el 59%")
        print("     con datos reales es suficiente para justificar su uso.")
    else:
        print(f"  ⚠️  Aportación moderada ({delta_f1:+.3f} F1).")
        print("     Considerar si el artefacto vale la mejora.")

    return delta_f1


# ---------------------------------------------------------------------------
# Recomendación final
# ---------------------------------------------------------------------------

def recomendacion(df_raw: pd.DataFrame, delta_f1: float) -> None:
    _separador("4. Recomendación")

    df_raw = df_raw.copy()
    df_raw["sin_datos"] = df_raw["sl_num_conciertos"].isna().astype(int)
    contingencia = pd.crosstab(df_raw["nivel"], df_raw["sin_datos"])
    _, p, _, _ = stats.chi2_contingency(contingencia)

    nan_aleatorio = p >= 0.05
    impacto_bajo  = abs(delta_f1) < 0.02

    if nan_aleatorio and impacto_bajo:
        print("  ✅ VALIDACIÓN SUPERADA")
        print("     El NaN es aleatorio y el impacto en el modelo es bajo.")
        print("     La imputación a 0 es aceptable. Mantener la configuración actual.")

    elif not nan_aleatorio and not impacto_bajo:
        print("  ❌ ARTEFACTO PROBABLE — ACCIÓN RECOMENDADA")
        print("     El NaN está correlacionado con el tier Y las features de setlist.fm")
        print("     tienen impacto significativo en el modelo.")
        print("     Opciones:")
        print("       A) Añadir feature binaria 'sl_tiene_datos' (0/1) para que el")
        print("          modelo distinga NaN real de 0 real.")
        print("       B) Imputar con la mediana del tier en lugar de 0.")
        print("       C) Excluir sl_num_conciertos y sl_num_paises del modelo final.")

    elif not nan_aleatorio and impacto_bajo:
        print("  ⚠️  RIESGO BAJO — DOCUMENTAR")
        print("     El NaN está correlacionado con el tier, pero el impacto en el")
        print("     modelo es pequeño. El artefacto existe pero no es crítico.")
        print("     Recomendación: documentar en el TFM y añadir feature 'sl_tiene_datos'")
        print("     como mejora futura.")

    else:
        print("  ✅ SEÑAL REAL — MANTENER")
        print("     El NaN es aleatorio y las features de setlist.fm tienen señal real.")
        print("     Mantener la configuración actual sin cambios.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Validación de la imputación de setlist.fm")
    print(f"Dataset: {CSV_PATH}")

    if not CSV_PATH.exists():
        print(f"\n❌ No se encuentra: {CSV_PATH}")
        print("   Ejecuta primero: python -m src.features.build_features")
        return

    df_raw = pd.read_csv(CSV_PATH)
    print(f"Shape: {df_raw.shape}")

    analisis_nan_vs_tier(df_raw)
    analisis_distribucion_con_datos(df_raw)
    delta_f1 = analisis_impacto_modelo()
    recomendacion(df_raw, delta_f1)

    _separador()
    print("  Análisis completado.")


if __name__ == "__main__":
    main()
