#Funciones de negocio: ganancia, umbral de decisión y métricas de apoyo.

import numpy as np
import pandas as pd


def ganancia(fraude, monto, rechazar, margen=0.25):
    # Ganancia total de una política de aprobación.
    # Por cada transacción aprobada: si es legítima se gana margen * monto
    # si es fraude se pierde el monto completo. Las rechazadas no suman ni restan.

    fraude = np.asarray(fraude)
    monto = np.asarray(monto)
    aprobar = ~np.asarray(rechazar, dtype=bool)

    legitimas_aprobadas = aprobar & (fraude == 0)
    fraudes_aprobados = aprobar & (fraude == 1)
    return margen * monto[legitimas_aprobadas].sum() - monto[fraudes_aprobados].sum()


def umbral_teorico(margen=0.25):
    # Probabilidad de fraude a partir de la cual conviene rechazar: margen / (1 + margen).
    return margen / (1 + margen)


def buscar_umbral(fraude, monto, puntaje, umbrales, margen=0.25):
    # Calcula la ganancia de cada umbral candidato.
    # Se rechaza una transacción si su puntaje es mayor o igual al umbral.
    # Sirve tanto para probabilidades (0 a 1) como para el score actual (0 a 100).
    
    puntaje = np.asarray(puntaje)
    resultados = [(umbral, ganancia(fraude, monto, puntaje >= umbral, margen)) for umbral in umbrales]
    return pd.DataFrame(resultados, columns=["umbral", "ganancia"])


def metricas_negocio(fraude, monto, rechazar, margen=0.25):
    # Ganancia y métricas de apoyo

    fraude = np.asarray(fraude)
    monto = np.asarray(monto)
    rechazar = np.asarray(rechazar, dtype=bool)

    monto_fraude_total = monto[fraude == 1].sum()
    monto_rechazado = monto[rechazar].sum()
    monto_fraude_rechazado = monto[rechazar & (fraude == 1)].sum()

    return {
        "ganancia": ganancia(fraude, monto, rechazar, margen),
        "pct_monto_fraude_evitado": monto_fraude_rechazado / monto_fraude_total * 100,
        "precision_monto_rechazadas": (
            monto_fraude_rechazado / monto_rechazado * 100 if monto_rechazado > 0 else np.nan
        ),
        "tasa_aprobacion": (~rechazar).mean() * 100,
    }