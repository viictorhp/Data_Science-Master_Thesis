"""
test_registry.py
================
Tests para config/registry.py (cargar / guardar).
No necesita APIs ni ficheros externos — usa ficheros temporales.
"""

import pytest
import pandas as pd
from pathlib import Path

from config.registry import COLUMNS, cargar, guardar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_minimo(tmp_path: Path) -> Path:
    """CSV mínimo con 3 artistas de prueba."""
    path = tmp_path / "artistas_registry.csv"
    df = pd.DataFrame([
        {
            "nombre_canonico": "Artista A",
            "spotify_id": "id_a", "spotify_nombre": "Artista A", "spotify_score": "1.0",
            "lastfm_nombre": "Artista A", "lastfm_mbid": None, "lastfm_score": "1.0",
            "yt_channel_id": "UC_aaa", "yt_nombre_canal": "Artista A", "yt_score": "1.0",
            "sl_nombre": "Artista A", "sl_mbid": "mbid_a", "sl_score": "1.0",
            "manual_override": "True", "notas": "",
        },
        {
            "nombre_canonico": "Artista B",
            "spotify_id": "id_b", "spotify_nombre": "Artista B", "spotify_score": "0.9",
            "lastfm_nombre": "Artista B", "lastfm_mbid": None, "lastfm_score": "0.9",
            "yt_channel_id": None, "yt_nombre_canal": None, "yt_score": None,
            "sl_nombre": None, "sl_mbid": None, "sl_score": None,
            "manual_override": "False", "notas": "sin canal",
        },
        {
            "nombre_canonico": "Artista C",
            "spotify_id": "id_c", "spotify_nombre": "Artista C", "spotify_score": "1.0",
            "lastfm_nombre": "Artista C", "lastfm_mbid": None, "lastfm_score": "1.0",
            "yt_channel_id": "UC_aaa",  # mismo canal que Artista A
            "yt_nombre_canal": "Artista A",
            "yt_score": "1.0",
            "sl_nombre": None, "sl_mbid": None, "sl_score": None,
            "manual_override": "True", "notas": "canal compartido con Artista A",
        },
    ])
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests de cargar()
# ---------------------------------------------------------------------------

def test_cargar_devuelve_dataframe(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_cargar_columnas_correctas(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    for col in COLUMNS:
        assert col in df.columns, f"Falta columna: {col}"


def test_cargar_manual_override_es_bool(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    assert df["manual_override"].dtype == bool
    assert df.loc[df["nombre_canonico"] == "Artista A", "manual_override"].iloc[0] == True
    assert df.loc[df["nombre_canonico"] == "Artista B", "manual_override"].iloc[0] == False


def test_cargar_fichero_inexistente_lanza_error(tmp_path, monkeypatch):
    monkeypatch.setattr("config.registry.REGISTRY_PATH", tmp_path / "no_existe.csv")
    with pytest.raises(FileNotFoundError):
        cargar()


def test_cargar_campo_nulo_permanece_nulo(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    b = df.loc[df["nombre_canonico"] == "Artista B"]
    assert pd.isna(b["yt_channel_id"].iloc[0])
    assert pd.isna(b["sl_mbid"].iloc[0])


# ---------------------------------------------------------------------------
# Tests de guardar()
# ---------------------------------------------------------------------------

def test_guardar_crea_fichero(tmp_path, monkeypatch):
    out = tmp_path / "sub" / "registry.csv"
    monkeypatch.setattr("config.registry.REGISTRY_PATH", out)
    df = pd.DataFrame([{"nombre_canonico": "Test"}])
    guardar(df)
    assert out.exists()


def test_guardar_produce_columnas_schema(tmp_path, monkeypatch):
    out = tmp_path / "registry.csv"
    monkeypatch.setattr("config.registry.REGISTRY_PATH", out)
    df = pd.DataFrame([{"nombre_canonico": "Test", "spotify_id": "sp1"}])
    guardar(df)
    leido = pd.read_csv(out)
    for col in COLUMNS:
        assert col in leido.columns


def test_guardar_no_incluye_columnas_extra(tmp_path, monkeypatch):
    out = tmp_path / "registry.csv"
    monkeypatch.setattr("config.registry.REGISTRY_PATH", out)
    df = pd.DataFrame([{"nombre_canonico": "Test", "columna_extra": "ignorar"}])
    guardar(df)
    leido = pd.read_csv(out)
    assert "columna_extra" not in leido.columns


# ---------------------------------------------------------------------------
# Tests de round-trip cargar / guardar
# ---------------------------------------------------------------------------

def test_round_trip_preserva_datos(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df_orig = cargar()
    guardar(df_orig)
    df_back = cargar()
    assert list(df_back["nombre_canonico"]) == list(df_orig["nombre_canonico"])


def test_round_trip_manual_override_sigue_siendo_bool(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    guardar(df)
    df2 = cargar()
    assert df2["manual_override"].dtype == bool


# ---------------------------------------------------------------------------
# Tests del modelo de datos (canales compartidos)
# ---------------------------------------------------------------------------

def test_canal_compartido_permitido(tmp_path, monkeypatch):
    """Dos artistas pueden tener el mismo yt_channel_id."""
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    compartidos = df.groupby("yt_channel_id")["nombre_canonico"].count()
    assert compartidos["UC_aaa"] == 2


def test_nombre_canonico_unico(tmp_path, monkeypatch):
    path = _csv_minimo(tmp_path)
    monkeypatch.setattr("config.registry.REGISTRY_PATH", path)
    df = cargar()
    assert not df["nombre_canonico"].duplicated().any()
