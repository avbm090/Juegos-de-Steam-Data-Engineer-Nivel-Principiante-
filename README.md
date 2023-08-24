# Proyecto de Machine Learning (Soy Henry): Juegos de Steam

## Descripción

En este proyecto presentamos los resultados de búsqueda de una base de datos de videojuegos de la plataforma Steam a través de la plataforma Render. El proyecto se enmarca en la institución Soy Henry y tiene como objetivo brindar una primera aproximación al trabajo de un data engineer.

## Funcionalidad

### Búsqueda

La aplicación de búsqueda proporciona resultados relevantes basados en varios criterios y preferencias. Los juegos pueden ser encontrados por título, género, desarrollador y otros criterios. Cada resultado de búsqueda incluye información relevante del año de lanzamiento.

### Métodos de machine learning utilizados y filtrado de datos

A la hora de codificar los datos para implementarlos en algún algoritmo, se consideró etiquetar los datos con "one hot encoding", "integer encoding" o "get dummies" . Se consideró que get dummies sería la mejor opción, pero no se pudo utilizar por limitaciones del modelo de cómputo del procesador utilizado. Se optó por utilizar one hot encoding.

Para la predicción de precios, se compararon varios algoritmos de machine learning, incluyendo regresión polinomial, decision tree, random forest, gradient boosting y support vector regression (SVR). También se exploró la posibilidad de utilizar algoritmos como k-means y PCA. El primero se usó para generar nuevas etiquetas que faciliten la implementación de algoritmos de regresión, mientras que el segundo se aplicó para reducir la dimensionalidad considerando las características más relevantes. Se pudo evidenciar con las métricas obtenidas, que dichos modelos no eran adecuados en este caso para este conjunto de datos. Para el algoritmo PCA se midió el nivel de significancia de las componentes halladas utilizando el ratio acumulativo de varianza. Los valores obtenidos no permitieron poder descartar muchas componentes principales, se debían eliminar 3 para poder explicar aproximadamente el 85% de la variabilidad total de los datos. Quedó por considerar utilizar k-nearest neighbors, de dos formas: una para generar nuevos agrupamientos con la intención de posteriormente utilizar algún algoritmo de regresión, o directamente utilizarlo como regresor. De la comparación entre modelos, se observó que el método gradient boosting presentó las mejores métricas, el valor del RMSI presentó magnitud inferiores a las del rango intercuartil, pero sería deseable que, dicho valor sea significativamente más chico que el rango inter cuartil. Se  considera que este valor se puede mejorar, por ejemplo, utilizando más valores en las grillas utilizadas . Quedó pendiente de probar el método SVR con otros núcleos, dado que, la forma que presentaba la dispersión de los datos sugería la posible implementación de un hiperplano generado por un núcleo radial,pero por falta de tiempo, esto no se pudo corroborar. 

## Archivos

- **etl:** Esta etapa muestra el procesamiento inicial de los datos, incluyendo cambios en los formatos de las columnas originales del dataframe y eliminación de columnas consideradas irrelevantes para el proyecto.
- **eda_ml:** En este archivo se detalla y justifica el proceso de Exploratory Data Analysis (EDA), así como la elección del algoritmo de machine learning.
- **eda_ml.pdf:** Una versión en formato PDF del archivo anterior para una visualización rápida.
- **main:** El archivo principal que contiene las funciones presentadas en la plataforma Render.
- **steam_games.json:** La base de datos original que contiene la lista de características de los juegos.
- **steam_games.parquet:** Este archivo se generó utilizando el código escrito en etl.py.
- **gradient_boosting:** En este archivo se entrena el dataset limpio con el algoritmo gradient boosting.

## Tecnología Utilizada
Python | FastAPI | Pandas | Scikit-Learn | Render

