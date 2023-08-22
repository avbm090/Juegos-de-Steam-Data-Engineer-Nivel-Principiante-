# Proyecto de Machine Learning (Soy Henry): Juegos de Steam

## Descripción

En este proyecto presentamos los resultados de búsqueda de una base de datos de videojuegos de la plataforma Steam a través de la plataforma Render. El proyecto se enmarca en la institución Soy Henry y tiene como objetivo brindar una primera aproximación al trabajo de un data engineer.

## Funcionalidad

### Búsqueda

La aplicación de búsqueda proporciona resultados relevantes basados en varios criterios y preferencias. Los juegos pueden ser encontrados por título, género, desarrollador y otros criterios. Cada resultado de búsqueda incluye información relevante del año de lanzamiento.

### Métodos de machine learning utilizados

Para la predicción de precios, se compararon varios algoritmos de machine learning, incluyendo regresión polinomial, decision tree, random forest, gradient boosting y support vector regression (SVR). También se exploró la posibilidad de utilizar algoritmos como k-means y PCA. El primero se usó para generar nuevas etiquetas que faciliten la implementación de algoritmos de regresión, mientras que el segundo se aplicó para reducir la dimensionalidad considerando las características más relevantes. Se pudo evidenciar con las métricas obtenidas, que dichos modelos para reducir dimensionalidad no eran adecuados para este conjunto de datos. Por otro lado, el método gradient boosting presentó las mejores métricas, el valor del RMSI tenía dimensiones inferiores a las del rango intercuartil, pero se considera que esto se puede mejorar, por ejemplo, utilizando más valores en las grillas utilizadas . Quedó pendiente de probar el método SVR con núcleo radial.    Luego se intentó aplicar algún algoritmo de regresión sobre los componentes principales hallados.

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

