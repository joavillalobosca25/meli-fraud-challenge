"""Entrena los modelos candidatos y los guarda como paquetes listos para usar.
Uso - desde la raíz del proyecto: python -m src.train
"""

import platform
from datetime import datetime

import joblib
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from src.business import metricas_negocio, umbral_teorico
from src.data import RAIZ, cargar_config, cargar_datos, separar_temporal
from src.features import PreparadorVariables

ESCENARIOS = {
    "modelo_sin_score": False,  # El modelo reemplaza al score actual
    "modelo_con_score": True,   # El modelo complementa al score actual
}


def entrenar_paquete(entrenamiento, validacion, config, usar_score):
    """Entrena el modelo candidato y lo empaqueta con su preparación, su umbral y sus metadatos."""
    parametros = config["modelo"]["hiperparametros"]
    margen = config["margen_ganancia"]
    objetivo = config["target"]

    preparador = PreparadorVariables(
        usar_score=usar_score,
        codificacion_j=config["modelo"]["codificacion_j"],
        semilla=config["seed"],
    )
    X_entrenamiento = preparador.fit_transform(entrenamiento)
    X_validacion = preparador.transform(validacion)

    modelo = xgb.XGBClassifier(
        **parametros,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=config["seed"],
    )
    # El early stopping vigila validación, igual que en el notebook 03, para reproducir el mismo modelo
    modelo.fit(
        X_entrenamiento, entrenamiento[objetivo],
        eval_set=[(X_validacion, validacion[objetivo])],
        verbose=False,
    )

    umbral = umbral_teorico(margen)
    proba_validacion = modelo.predict_proba(X_validacion)[:, 1]
    metricas = metricas_negocio(
        validacion[objetivo], validacion["monto"], rechazar=proba_validacion >= umbral, margen=margen
    )

    metadatos = {
        "escenario": "con score" if usar_score else "sin score",
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "periodo_entrenamiento": [str(entrenamiento["fecha"].min()), str(entrenamiento["fecha"].max())],
        "periodo_validacion": [str(validacion["fecha"].min()), str(validacion["fecha"].max())],
        "variables": list(X_entrenamiento.columns),
        "codificacion_j": config["modelo"]["codificacion_j"],
        "hiperparametros": parametros,
        "arboles": modelo.best_iteration + 1,
        "margen_ganancia": margen,
        "umbral": umbral,
        "metricas_validacion": {
            "log_loss": float(log_loss(validacion[objetivo], proba_validacion)),
            "auc": float(roc_auc_score(validacion[objetivo], proba_validacion)),
            **{nombre: float(valor) for nombre, valor in metricas.items()},
        },
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    return {"preparador": preparador, "modelo": modelo, "umbral": umbral, "metadatos": metadatos}


def main():
    """Entrena los dos escenarios con entrenamiento y validación, y guarda cada paquete."""
    config = cargar_config()
    datos = cargar_datos(config)

    # Test no se usa aquí: se reserva para evaluar los paquetes en el notebook 04
    entrenamiento, validacion, _ = separar_temporal(datos, config)

    carpeta = RAIZ / config["modelo"]["carpeta_salida"]
    carpeta.mkdir(exist_ok=True)

    for nombre, usar_score in ESCENARIOS.items():
        paquete = entrenar_paquete(entrenamiento, validacion, config, usar_score)
        ruta = carpeta / f"{nombre}.joblib"
        joblib.dump(paquete, ruta)

        metricas = paquete["metadatos"]["metricas_validacion"]
        print(
            f"{nombre}: {paquete['metadatos']['arboles']} árboles | "
            f"log-loss {metricas['log_loss']:.4f} | ganancia en validación {metricas['ganancia']:,.0f} | "
            f"guardado en {ruta.relative_to(RAIZ)}"
        )


if __name__ == "__main__":
    main()