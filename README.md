# Proyecto de Data engineering (Soy Henry): Juegos de Steam

## Descripción

En este proyecto se presentan los resultados de recolección de datos, de una API de videojuegos de la plataforma Steam a través de Render. El proyecto tiene como objetivo brindar una primera aproximación al trabajo de un data engineer.

## Funcionalidad

### Búsqueda

La aplicación de búsqueda proporciona resultados relevantes basados en varios criterios y preferencias. Los juegos pueden ser encontrados por título, género, desarrollador y otros criterios. Cada resultado de búsqueda incluye información relevante del año de lanzamiento.

### Métodos de machine learning utilizados y filtrado de datos

A la hora de codificar los datos para implementarlos en algún algoritmo, se consideró entre etiquetar los datos con "label encoding", "integer encoding" o utilizar directamente "get dummies" . En este caso, get dummies sería la mejor opción, pero no se pudo utilizar por limitaciones del modelo de cómputo del procesador utilizado. Se optó por utilizar label encoding.

Para la predicción de precios, se compararon varios algoritmos de machine learning, incluyendo regresión polinomial, decision tree, random forest, gradient boosting y support vector regression (SVR). Se descartó la posibilidad de utilizar regresión lineal de una forma muy simple, se corroboró que ninguna de las columnas presentaba una distribución normal, por lo que, por precaución y para evitar posible sesgo en los valores de los coeficientes de regresión se optó por descartar la posible aplicación de método ( aunque se comprende que una manera mas estricta de confirmar esto, podria ser confirmando que se cumplan las hipótesis del análisis de residuos, entre otras) . También se exploró la posibilidad de utilizar algoritmos como k-means y PCA. El primero se usó para generar nuevas etiquetas que faciliten la implementación de algoritmos de regresión, mientras que el segundo se aplicó para reducir la dimensionalidad considerando las características más relevantes. Se pudo evidenciar con las métricas obtenidas, que dichos modelos no eran adecuados en este caso para este conjunto de datos, esto tambien se podria deber al tioo de codificación que se utilizó. Para el algoritmo PCA se midió el nivel de significancia de las componentes halladas utilizando el ratio acumulativo de varianza. Los valores obtenidos no permitieron poder descartar muchas componentes principales, se debían eliminar 2 para poder explicar aproximadamente el 85% de la variabilidad total de los datos. Quedó por considerar utilizar k-nearest neighbors, de dos formas: una para generar nuevos agrupamientos con la intención de posteriormente utilizar algún algoritmo de regresión, y la otra forma sería utilizarlo directamente como regresor. Para la implementación de k-means, se pensó en agrupar datos para conseguir nuevas etiquetas que puedan ayudar a mejorar las predicciones. Se midió la efectividad del métido con dos métricas, la inercia y silhouette score. Los resultados no fueron positivos para implementar este método. En la fase de comparación entre modelos, se observó que el método gradient boosting presentó las mejores métricas, el valor del RMSI presentó magnitud inferiores a las del rango intercuartil, pero sería deseable que dicha relación sea significativamente más chica. Tambien se corroboró que con este modelo se obtuvo un mejor balance de sub y sobre ajuste respecto a los otros. si bien también se utilizó validación cruzada para reducir el sesgo de las metricas obtenidas, se observó que estos valores se podrían mejorar, por ejemplo, utilizando más valores en las grillas utilizadas . Quedó pendiente de probar el método SVR con otros núcleos, dado que, la forma que presentaba la dispersión de los datos no descartaba la posibilidad de utilizar núcleo radial ,pero por falta de tiempo, esto no se pudo corroborar. 

## Archivos

- **etl:** Esta etapa muestra el procesamiento inicial de los datos, incluyendo cambios en los formatos de las columnas originales del dataframe y eliminación de columnas consideradas irrelevantes para el proyecto.
- **eda_ml:** En este archivo se detalla y justifica el proceso de Exploratory Data Analysis (EDA), así como la elección del algoritmo de machine learning.
- **main:** El archivo principal que contiene las funciones presentadas en la plataforma Render.
- **steam_games.json:** La base de datos original que contiene la lista de características de los juegos.
- **steam_games.parquet:** Este archivo se generó utilizando el código escrito en etl.py.
- **gradient_boosting:** En este archivo se entrena el dataset limpio con el algoritmo gradient boosting.

## Tecnología Utilizada
Python | FastAPI | Pandas | Scikit-Learn | Render

