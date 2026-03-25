import pandas as pd
from sklearn.model_selection import train_test_split

from models import AlcoholLogisticModel

def run_test():
    # Crear dataframe con datos inventados (para probar que funciona)
    df = pd.DataFrame({
        "rango_edad": ["18-24", "25-34", "35-44", "18-24", "45-54", "55-64", "25-34", "35-44"],
        "tipo_accidente": ["Alcance", "Caída", "Atropello a persona", "Colisión frontal", "Alcance", "Caída", "Otro", "Alcance"],
        "estado_meteorológico": ["Despejado", "Lluvia débil", "Despejado", "Nublado", "Despejado", "Nublado", "Lluvia débil", "Despejado"],
        "tipo_vehiculo_grupo": ["Coches", "Motos", "Coches", "Coches", "Motos", "Coches", "Coches", "Motos"],
        "tipo_persona": ["Conductor", "Conductor", "Peatón", "Conductor", "Conductor", "Peatón", "Conductor", "Conductor"],
        "sexo": ["Hombre", "Mujer", "Mujer", "Hombre", "Hombre", "Mujer", "Hombre", "Mujer"],
        "hora_int": [23, 2, 18, 9, 1, 15, 22, 3],
        "dia_sem": [5, 6, 2, 1, 5, 3, 4, 6],
        "positiva_alcohol": [1, 1, 0, 0, 1, 0, 0, 1],
    })

    # Definir las variables predictoras (X)
    X = df[['rango_edad','tipo_accidente', 'estado_meteorológico', 'tipo_vehiculo_grupo', 'tipo_persona', 'sexo', 'hora_int','dia_sem']].copy()

    # Definir la variable objetivo (y)
    y = df['positiva_alcohol'].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Definir columnas por tipo
    categorical_f = ['rango_edad', 'tipo_accidente', 'estado_meteorológico', 'tipo_vehiculo_grupo', 'tipo_persona', 'sexo']
    numeric_f = ['hora_int', 'dia_sem']

    # Entrenar modelo
    model = AlcoholLogisticModel(
        categorical_f = categorical_f,
        numeric_f = numeric_f,
        random_state = 42,
        max_iter = 1000,
        class_weight = "balanced"
    )
    model.fit(X_train,y_train)

    # Evaluar/predecir
    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Checks 
    assert len(y_pred) == len(X_test), "La longitud de y_pred no coincide con X_test"
    assert proba.min() >= 0 and proba.max() <= 1, "Las probabilidades deben estar en [0, 1]"
    assert "accuracy" in metrics and "f1" in metrics and "confusion_matrix" in metrics, "Faltan métricas en evaluate()"

    print("Test OK")
    print("Métricas:",{k: v for k, v in metrics.items() if k != "confusion_matrix"})
    print("Confusion matrix:\n", metrics['confusion_matrix'])

if __name__ == "__main__":
    run_test()
