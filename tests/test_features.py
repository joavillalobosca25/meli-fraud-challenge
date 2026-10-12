"""Pruebas del preparador de variables, en especial la codificación de j."""

import pandas as pd
import pytest

from src.features import PreparadorVariables


def datos_ejemplo():
    """Diez transacciones mínimas con todas las columnas que usa el preparador."""
    return pd.DataFrame({
        "b": [0.5] * 10, "c": [1.0] * 10, "d": [1.0] * 10, "e": [0.1] * 10, "f": [1.0] * 10,
        "h": [1] * 10, "l": [1.0] * 10, "m": [1.0] * 10, "monto": [10.0] * 10,
        "fecha": pd.to_datetime(["2020-03-10 10:00:00"] * 10),
        "n": [1] * 10,
        "p": ["Y", "N"] * 5,
        "a": [4] * 10,
        "g": ["BR"] * 10,
        "o": ["Y"] * 10,
        "j": ["cat_a"] * 5 + ["cat_b"] * 5,
        "score": [50] * 10,
        "fraude": [1, 0] * 5,
    })


def test_opcion_invalida():
    # Una opción mal escrita debe producir un error, no un modelo sin la variable j
    with pytest.raises(ValueError):
        PreparadorVariables(codificacion_j="otra")


def test_columnas_segun_codificacion():
    datos = datos_ejemplo()

    solo_frecuencia = PreparadorVariables(codificacion_j="frecuencia").fit_transform(datos)
    assert "j_frecuencia" in solo_frecuencia.columns
    assert "j_tasa" not in solo_frecuencia.columns

    solo_tasa = PreparadorVariables(codificacion_j="tasa").fit_transform(datos)
    assert "j_tasa" in solo_tasa.columns
    assert "j_frecuencia" not in solo_tasa.columns

    ambas = PreparadorVariables(codificacion_j="ambas").fit_transform(datos)
    assert {"j_frecuencia", "j_tasa"} <= set(ambas.columns)


def test_score_opcional():
    datos = datos_ejemplo()
    assert "score" not in PreparadorVariables(usar_score=False).fit_transform(datos).columns
    assert "score" in PreparadorVariables(usar_score=True).fit_transform(datos).columns


def test_categoria_nueva_recibe_valores_por_defecto():
    # Simula producción: una categoría de j y un país que no existían en entrenamiento
    datos = datos_ejemplo()
    preparador = PreparadorVariables(codificacion_j="ambas", min_transacciones_pais=1).fit(datos)
    nueva = datos.head(1).assign(j="cat_nueva", g="ZZ")
    resultado = preparador.transform(nueva)

    assert resultado["j_frecuencia"].iloc[0] == 0
    assert resultado["g"].iloc[0] == "otros"
    assert resultado["j_tasa"].notna().iloc[0]


def test_p_desconocido_queda_nulo():
    # Un valor de p distinto de Y o N no debe convertirse en 0 sin aviso
    datos = datos_ejemplo()
    preparador = PreparadorVariables().fit(datos)
    resultado = preparador.transform(datos.head(1).assign(p="y"))
    assert resultado["p"].isna().iloc[0]