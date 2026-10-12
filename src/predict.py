"""Uso de un modelo candidato empaquetado: cargarlo y decidir sobre transacciones nuevas."""

import joblib


def cargar_paquete(ruta):
    """Carga un paquete guardado por src/train.py.
    """
    return joblib.load(ruta)


def predecir(paquete, datos):
    """Devuelve la probabilidad de fraude y la decisión de rechazo de cada transacción.
    Usa el preparador guardado en el paquete, así las variables se construyen exactamente
    igual que en el entrenamiento. No necesita la columna de fraude.
    """
    variables = paquete["preparador"].transform(datos)
    probabilidad = paquete["modelo"].predict_proba(variables)[:, 1]
    rechazar = probabilidad >= paquete["umbral"]
    return probabilidad, rechazar