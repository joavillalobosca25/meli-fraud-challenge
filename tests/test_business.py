#Pruebas de las funciones de negocio

import pytest

from src.business import buscar_umbral, ganancia, metricas_negocio, umbral_teorico

# Cuatro transacciones de 100: dos legítimas y dos fraudes
FRAUDE = [0, 0, 1, 1]
MONTO = [100, 100, 100, 100]


def test_aprobar_todo():
    # Gana 25 + 25 por las legítimas y pierde 100 + 100 por los fraudes
    assert ganancia(FRAUDE, MONTO, rechazar=[False] * 4) == -150


def test_rechazar_solo_fraudes():
    # Gana las dos legítimas y evita ambos fraudes
    assert ganancia(FRAUDE, MONTO, rechazar=[False, False, True, True]) == 50


def test_rechazar_todo():
    assert ganancia(FRAUDE, MONTO, rechazar=[True] * 4) == 0


def test_umbral_teorico():
    assert umbral_teorico(0.25) == pytest.approx(0.2)


def test_buscar_umbral():
    # Con umbral 0,5 se rechazan las dos transacciones de puntaje 0,9, que son los fraudes
    resultado = buscar_umbral(FRAUDE, MONTO, puntaje=[0.1, 0.1, 0.9, 0.9], umbrales=[0.5])
    assert resultado.loc[0, "ganancia"] == 50


def test_metricas_rechazo_perfecto():
    metricas = metricas_negocio(FRAUDE, MONTO, rechazar=[False, False, True, True])
    assert metricas["pct_monto_fraude_evitado"] == 100
    assert metricas["precision_monto_rechazadas"] == 100
    assert metricas["tasa_aprobacion"] == 50