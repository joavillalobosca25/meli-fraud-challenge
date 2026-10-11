#Construcción de las variables del modelo a partir de los datos iniciales

import pandas as pd

NUMERICAS = ["b", "c", "d", "e", "f", "h", "l", "m", "monto"]
CATEGORICAS = ["a", "g", "o"]


class PreparadorVariables:
    # Aplica las decisiones del EDA. Aprende solo de entrenamiento (fit) y transforma cualquier conjunto.

    def __init__(self, min_transacciones_pais=100, usar_score=False):
        self.min_transacciones_pais = min_transacciones_pais
        self.usar_score = usar_score

    def fit(self, datos):
        # Aprende de entrenamiento los países frecuentes, las frecuencias de j y las categorías válidas.
        conteo_paises = datos["g"].value_counts()
        self.paises_frecuentes_ = conteo_paises[conteo_paises >= self.min_transacciones_pais].index
        self.frecuencia_j_ = datos["j"].value_counts()

        base = self._categoricas_base(datos)
        self.categorias_ = {columna: sorted(base[columna].unique()) for columna in CATEGORICAS}
        return self

    def transform(self, datos):
        # Construye la tabla de variables del modelo.
        variables = datos[NUMERICAS].copy()

        variables["hora"] = datos["fecha"].dt.hour
        variables["n"] = datos["n"]
        variables["p"] = datos["p"].map({"Y": 1, "N": 0})

        # Categorías nuevas que no estaban en entrenamiento quedan como nulas
        base = self._categoricas_base(datos)
        for columna in CATEGORICAS:
            variables[columna] = pd.Categorical(base[columna], categories=self.categorias_[columna])

        # j por frecuencia en entrenamiento; una categoría nueva tiene frecuencia 0
        variables["j_frecuencia"] = datos["j"].map(self.frecuencia_j_).fillna(0)

        if self.usar_score:
            variables["score"] = datos["score"]
        return variables

    def _categoricas_base(self, datos):
        # Prepara a, g y o como texto, con los nulos y países poco frecuentes agrupados.
        pais = datos["g"].where(datos["g"].isin(self.paises_frecuentes_), "otros")
        return pd.DataFrame({
            "a": datos["a"].astype(str),
            "g": pais.mask(datos["g"].isna(), "nulo"),
            "o": datos["o"].fillna("nulo"),
        })