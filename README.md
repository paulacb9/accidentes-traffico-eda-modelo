# Predicción de positivos en alcohol en accidentes de tráfico

## Descripción del proyecto

Este proyecto realiza un **Análisis Exploratorio de Datos (EDA)** sobre un dataset de accidentalidad y desarrolla un **modelo de regresión logística** para predecir la probabilidad de que un accidente esté asociado a un positivo en alcohol.

El objetivo es identificar patrones relevantes y construir un modelo interpretable que permita entender los factores asociados.

---

## Estructura del proyecto

```
.
├── ProyectoEDA_paula.ipynb   # Notebook principal (EDA + modelo)
├── models.py                 # Clase del modelo (pipeline + lógica)
├── test_model.py             # Test básico del modelo
├── 2024_Accidentalidad.csv   # Dataset utilizado
├── requirements.txt          # Dependencias
└── README.md                 # Documentación del proyecto
```

---

## Tecnologías utilizadas

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib / Seaborn

---

## Metodología

### 1. Limpieza de datos

* Tratamiento de valores nulos
* Normalización de variables
* Creación de variables derivadas

### 2. Tratamiento de coordenadas

* Se imputan coordenadas faltantes cuando existe una **referencia interna consistente** (misma localización con coordenadas válidas en otras filas)
* En casos donde existen múltiples coordenadas para una misma localización sin criterio claro, los registros se eliminan para evitar ambigüedad y posibles sesgos
* En algunos casos puntuales se ha recurrido a geocodificación manual, lo que se considera una limitación del análisis

---

### 3. Modelado

Se utiliza un modelo de **Regresión Logística** con:

* Pipeline de preprocesado:

  * OneHotEncoder para variables categóricas
  * StandardScaler para variables numéricas
* Manejo de desbalanceo mediante `class_weight="balanced"`

---

### 4. Evaluación

El modelo se evalúa mediante:

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión

---

## Cómo ejecutar el proyecto (Anaconda)
1. Clonar el repositorio

	`git clone https://github.com/tu_usuario/tu_repo.git`
	`cd tu_repo`

2. Crear entorno virtual

	`conda create -n accidentes python=3.10`

3. Activar entorno
	
	`conda activate accidentes`

4. Instalar dependencias

	`pip install -r requirements.txt`

5. Ejecutar el notebook
	
	`jupyter notebook`

Abrir:

	ProyectoEDA_paula.ipynb
---

## Conclusión

El modelo permite identificar patrones relevantes en los accidentes asociados al consumo de alcohol, proporcionando una base para análisis más avanzados o aplicaciones en prevención y seguridad vial.

---

## Autor

Paula C. Blanch
