# Prevención de fraude: modelo de maximización de ganancia

Modelo de Machine Learning para decidir qué transacciones aprobar y cuáles rechazar, maximizando la
ganancia del negocio: cada transacción legítima aprobada deja un 25% de su monto, y cada fraude aprobado
pierde el 100%.

## Resultados clave

Evaluación final en test (13 al 21 de abril de 2020), con modelos, umbral y referencias fijas:

| Política | Ganancia | Frente al score actual | Fraude evitado (monto) | Compras legítimas rechazadas |
|---|---|---|---|---|
| Aprobar todo | 226.254 | −10,8% | 0% | 0 |
| Score actual (corte 86) | 253.579 | — | 49,1% | 2.223 |
| **XGBoost sin score** (reemplaza al score) | **264.371** | **+4,3%** | 52,8% | 1.250 |
| **XGBoost con score** (complementa al score) | **272.695** | **+7,5%** | 56,8% | 1.019 |

- Ambos modelos superan al score actual de forma consistente (en el 99,3% y el 100% de 1.000 simulaciones bootstrap).
- Evitan más dinero de fraude que el score actual y rechazan a casi la mitad de clientes legítimos.
- La regla de decisión (rechazar si la probabilidad de fraude es mayor o igual a 0,2) se deriva de la
  ganancia del negocio y se verificó en test.

## Estructura del proyecto

```
├── config.yaml                 # Parámetros y decisiones congeladas (semilla, cortes, hiperparámetros)
├── requirements.txt            # Versiones exactas de las librerías
├── data/raw/                   # Aquí va el dataset (no se versiona)
├── models/                     # Modelos candidatos empaquetados (generados por src/train.py)
├── notebooks/
│   ├── 01_validacion_datos.ipynb   # Validación de la base y separación temporal
│   ├── 02_EDA.ipynb                # Análisis exploratorio sobre entrenamiento
│   ├── 03_modelado.ipynb           # Modelos, ajuste y comparación (entrenamiento y validación)
│   └── 04_evaluacion_test.ipynb    # Evaluación final en test e interpretación
├── src/
│   ├── data.py                 # Carga de datos y separación temporal
│   ├── features.py             # Construcción de variables (fit en entrenamiento, transform en el resto)
│   ├── business.py             # Ganancia, umbral teórico y métricas de negocio
│   ├── train.py                # Entrena y empaqueta los modelos candidatos
│   └── predict.py              # Carga un modelo empaquetado y decide sobre transacciones nuevas
└── tests/                      # Pruebas automáticas de los módulos de negocio y de variables
```

## Cómo ejecutar

Probado con Python 3.14.6 en Windows.

**1. Crear el entorno e instalar las dependencias**

```bash
python -m venv .venv
.venv\Scripts\activate          # En Windows
# source .venv/bin/activate     # En Mac o Linux
pip install -r requirements.txt
```

**2. Ubicar el dataset**

Copiar el archivo `MercadoLibre Data Scientist Technical Challenge - Dataset.csv` en `data/raw/`
con el nombre `dataset.csv`.

**3. Verificar con las pruebas automáticas** (opcional)

```bash
python -m pytest
```

**4. Ejecutar los notebooks en orden**

Abrir la carpeta del proyecto en Jupyter o VS Code, seleccionar el entorno `.venv` y ejecutar
los notebooks del 01 al 04. Cada uno corre de principio a fin de forma independiente.

Los modelos empaquetados ya están en `models/`, así que el notebook 04 puede ejecutarse directamente.
Para regenerarlos desde cero:

```bash
python -m src.train
```

Los resultados son reproducibles: la semilla, los cortes temporales y los hiperparámetros están fijos en `config.yaml`.

## Decisiones principales

- **Separación temporal:** entrenamiento (8 de marzo a 5 de abril), validación (6 a 12 de abril) y test
  (13 a 21 de abril). Separar al azar mezclaría periodos y daría una evaluación optimista.
- **Métrica principal: ganancia.** El modelo se elige por la ganancia del negocio; las demás métricas
  (AUC, Gini, PR-AUC, precisión y recall) se reportan para describirlo.
- **Umbral de 0,2:** un fraude cuesta lo mismo que el margen de cuatro ventas legítimas, así que conviene
  rechazar cuando la probabilidad de fraude es de al menos 20%. Se usa porque las probabilidades del modelo
  están calibradas.
- **Modelo:** XGBoost, elegido entre regresión logística, LightGBM y XGBoost, con hiperparámetros ajustados
  por log-loss y comparación entre modelos con bootstrap pareado.
- **Dos escenarios:** sin score (el modelo reemplaza al sistema actual) y con score (lo complementa).

## Informe

El informe con el análisis completo y las respuestas a las preguntas del enunciado está en `reports/informe.pdf`.
