"""Construcción de las variables del modelo a partir de los datos crudos."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

NUMERICAS = ["b", "c", "d", "e", "f", "h", "l", "m", "monto"]
CATEGORICAS = ["a", "g", "o"]
CODIFICACIONES_J = ("frecuencia", "tasa", "ambas")


class PreparadorVariables:
    """Aplica las decisiones del EDA. Aprende solo de entrenamiento (fit) y transforma cualquier conjunto."""

    def __init__(self, min_transacciones_pais=100, usar_score=False, codificacion_j="frecuencia", semilla=20240205):
        if codificacion_j not in CODIFICACIONES_J:
            raise ValueError(f"codificacion_j debe ser 'frecuencia', 'tasa' o 'ambas', no '{codificacion_j}'")

        self.min_transacciones_pais = min_transacciones_pais
        self.usar_score = usar_score
        self.codificacion_j = codificacion_j
        self.semilla = semilla

    def fit(self, datos):
        """Aprende de entrenamiento los países frecuentes, las codificaciones de j y las categorías válidas."""
        conteo_paises = datos["g"].value_counts()
        self.paises_frecuentes_ = conteo_paises[conteo_paises >= self.min_transacciones_pais].index
        self.frecuencia_j_ = datos["j"].value_counts()

        base = self._categoricas_base(datos)
        self.categorias_ = {columna: sorted(base[columna].unique()) for columna in CATEGORICAS}
        # "otros" y "nulo" siempre existen en g, aunque entrenamiento no los tenga,
        # para que un país nuevo en producción caiga en "otros" y no quede como nulo
        self.categorias_["g"] = sorted(set(self.categorias_["g"]) | {"otros", "nulo"})

        if self.codificacion_j in ("tasa", "ambas"):
            # Ajuste cruzado en 5 partes que mantienen la proporción de fraudes
            particiones = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.semilla)
            self.codificador_j_ = TargetEncoder(target_type="binary", cv=particiones)
            self.codificador_j_.fit(datos[["j"]], datos["fraude"])

        return self

    def transform(self, datos):
        """Construye la tabla de variables del modelo con lo aprendido en entrenamiento."""
        variables = datos[NUMERICAS].copy()

        variables["hora"] = datos["fecha"].dt.hour
        variables["n"] = datos["n"]
        # Cualquier valor distinto de Y o N queda como nulo, en lugar de disfrazarse de N
        variables["p"] = datos["p"].map({"Y": 1, "N": 0})

        # Categorías que no estaban en entrenamiento quedan como nulas
        base = self._categoricas_base(datos)
        for columna in CATEGORICAS:
            variables[columna] = pd.Categorical(base[columna], categories=self.categorias_[columna])

        # j por frecuencia en entrenamiento; una categoría nueva tiene frecuencia 0
        if self.codificacion_j in ("frecuencia", "ambas"):
            variables["j_frecuencia"] = datos["j"].map(self.frecuencia_j_).fillna(0)

        # j por tasa de fraude suavizada; una categoría nueva recibe la tasa promedio
        if self.codificacion_j in ("tasa", "ambas"):
            variables["j_tasa"] = self.codificador_j_.transform(datos[["j"]]).ravel()

        if self.usar_score:
            variables["score"] = datos["score"]

        return variables

    def fit_transform(self, datos):
        """Aprende de entrenamiento y lo transforma, con ajuste cruzado para la tasa de j.

        Para entrenamiento, la tasa de fraude de j se calcula con ajuste cruzado: cada transacción
        se codifica con una tasa calculada sin ella, para evitar fuga de información.
        Validación y test se transforman con `transform`, que usa la tasa de todo entrenamiento.
        """
        self.fit(datos)
        variables = self.transform(datos)

        if self.codificacion_j in ("tasa", "ambas"):
            variables["j_tasa"] = self.codificador_j_.fit_transform(datos[["j"]], datos["fraude"]).ravel()

        return variables

    def _categoricas_base(self, datos):
        """Prepara a, g y o como texto, con los nulos y los países poco frecuentes agrupados."""
        pais = datos["g"].where(datos["g"].isin(self.paises_frecuentes_), "otros")
        return pd.DataFrame({
            "a": datos["a"].astype(str),
            "g": pais.mask(datos["g"].isna(), "nulo"),
            "o": datos["o"].fillna("nulo"),
        })