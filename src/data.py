"""Carga de datos y separación temporal del dataset."""

from pathlib import Path

import pandas as pd
import yaml

# Raíz del proyecto: este archivo está en src/, subimos una carpeta
RAIZ = Path(__file__).resolve().parents[1]


def cargar_config():
    # Lee los parámetros del proyecto desde config.yaml
    with open(RAIZ / "config.yaml") as archivo:
        return yaml.safe_load(archivo)


def cargar_datos(config):
    # Carga el dataset y convierte la fecha con formato correcto, Si alguna fecha no cumple el formato, se detiene con un error.

    df = pd.read_csv(RAIZ / config["data_path"])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d %H:%M:%S")
    return df


def separar_temporal(df, config):
    # Separa el dataset en entrenamiento, validación y test según las fechas del config
    corte_validacion = pd.Timestamp(config["split"]["validacion_desde"])
    corte_test = pd.Timestamp(config["split"]["test_desde"])

    entrenamiento = df[df["fecha"] < corte_validacion]
    validacion = df[(df["fecha"] >= corte_validacion) & (df["fecha"] < corte_test)]
    test = df[df["fecha"] >= corte_test]

    assert len(entrenamiento) + len(validacion) + len(test) == len(df), (
        "Se perdieron transacciones en la separación"
    )
    return entrenamiento, validacion, test