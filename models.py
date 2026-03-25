import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

class AlcoholLogisticModel:
    """
    Modelo de regresión logística para predecir positivos en alcohol.
    Incluye preprocesado automático mediante Pipeline.
    """

    def __init__(self, categorical_f, numeric_f, random_state=42, max_iter=1000, class_weight="balanced"):
        """Guardar: qué columnas son categóricas, numéricas, parámetros del modelo (random_state, max_iter..), y construir el pipeline"""

        self.categorical_f = categorical_f
        self.numeric_f = numeric_f
        self.random_state = random_state
        self.max_iter = max_iter # Al usar OneHot, class_weight y tener datos desbalanceados, incrementamos el número máximo de iteraciones del algoritmo para asegurar la convergencia del modelo
        self.class_weight = class_weight # Importante para que el modelo aprenda bien, ya que nuestra variable objetivo es ~97%-> 0 y ~3%-> 1

        # Creamos el pipeline al inicializar el modelo
        self.pipeline = self._build_pipeline()


    def _build_pipeline(self):
        """Crea un pipeline con ColumnTransformer() y LogisticRegression()"""

        # Preprocesador : Numéricas -> StandardScaler, Categóricas -> OneHotEncoder
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_f),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_f) # handle_unknown para no dar error si entra una categoría desconocida
            ]
        )

        # Clasificador
        classifier = LogisticRegression(
            max_iter = self.max_iter,
            random_state = self.random_state,
            class_weight = self.class_weight
        )

        # Pipeline completo
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        return pipeline


    def fit(self, X_train, y_train):
        """Entrena el modelo con el conjunto de entrenamiento"""

        # Verificar que las columnas utilizadas por el modelo estén presentes en el conjunto de datos de entrada (columnas modelo - columnas entrada)
        missing = set(self.categorical_f + self.numeric_f) - set(X_train.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X_train: {missing}")

        self.pipeline.fit(X_train, y_train)
        return self


    def predict(self, X_test):
        """Devuelve la predicción de clase (0/1)."""

        # Asegurar que el modelo esta entrenado antes de predecir
        if self.pipeline is None:
            raise ValueError("El modelo no está entrenado. Llama primero a fit().")

        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test):
        """Devuelve la probabilidad estimada de pertenencia a cada clase."""

        if self.pipeline is None:
            raise ValueError("El modelo no está entrenado. Llama primero a fit().")

        return self.pipeline.predict_proba(X_test)


    def evaluate(self, X_test, y_test):
        """Evalúa el modelo y devuelve métricas de rendimiento"""

        # Predecir la clase (0/1) en el conjunto de test
        y_pred = self.predict(X_test)
        # Calcular métricas princiaples
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0), #zero_division para evitar problemas cuando divide por cero
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        # Devolver las métricas para usarlas en el notebook
        return metrics


    def save_model(self, path="alcohol_logistic_model.pkl"):
        """Guardar el modelo entrenado en un fichero."""

        with open(path, 'wb') as fp:
            pkl.dump(self.pipeline, fp)

    def load_model(self, path="alcohol_logistic_model.pkl"):
        """Carga un modelo previamente entrenado."""

        with open(path, 'rb') as fp:
            self.pipeline = pkl.load(fp)
        return self