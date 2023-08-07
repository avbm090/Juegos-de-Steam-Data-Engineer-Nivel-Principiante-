
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import fastparquet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Carga el modelo desde el archivo pickle
with open("gradient_boosting_model.pkl", 'rb') as file:
    loaded_data = pickle.load(file)
    
# Extraer el modelo y y_train del diccionario
loaded_model = loaded_data['model']
loaded_y_train = loaded_data['y_train']

# Se crea un objeto LabelEncoder
label_encoder = LabelEncoder()

# Se carga el dataframe
df = pd.read_parquet('steam_games.parquet')

app = FastAPI()


# Función para obtener los géneros más vendidos
@app.get("/genero/")
def genero(anio: int):
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    
    generos_count = {}
    for lista_generos in df_filtrado['genres']:
        if lista_generos is not None:
            for genero_juego in lista_generos:
                if genero_juego is not None:
                    generos_count[genero_juego] = generos_count.get(genero_juego, 0) + 1
    
    generos_top = dict(sorted(generos_count.items(), key=lambda item: item[1], reverse=True)[:5])
    return generos_top

# Función para obtener los juegos lanzados en un año
@app.get("/juegos/")
def juegos(anio: int):
   # Filtrar el DataFrame por el año deseado
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    juegos_unicos = df_filtrado["title"].unique()
    respuesta = {"juegos": juegos_unicos.tolist()}  # se convierte la lista en una lista Python
    return respuesta

# Función para obtener los specs más repetidos
@app.get("/specs/")
def specs(anio: int):
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    
    specs_count = {}
    for lista_specs in df_filtrado['specs']:
        if lista_specs is not None:
            for spec in lista_specs:
                specs_count[spec] = specs_count.get(spec, 0) + 1
    
    top_specs = dict(sorted(specs_count.items(), key=lambda item: item[1], reverse=True)[:5])
    return {"top_specs": list(top_specs.keys())}

# Función para obtener la cantidad de juegos lanzados en un año con early access
@app.get("/earlyacces/")
def earlyacces(anio: int):
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    frecuencias=df_filtrado["early_access"].value_counts().to_dict()
    
    return frecuencias

# Función para obtener el análisis de sentimiento

@app.get("/sentiment/")
def sentiment(anio: int):
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    frecuencia=df_filtrado["sentiment"].value_counts().to_dict()
    return frecuencia

# Función para obtener los juegos top 5 con mayor metascore
@app.get("/metascore/")
def metascore(anio: int):
    # Filtrar el DataFrame por el año proporcionado
    df_anio = df[df["release_date"]== anio]

    # Eliminar duplicados basados en la columna "title"
    juegos_unicos = df_anio.drop_duplicates(subset="title")

    # Ordenar los juegos únicos por Metascore de manera descendente
    juegos_ordenados = juegos_unicos.sort_values(by="metascore", ascending=False)

    # Tomar los 5 juegos top con los Metascores más altos
    top_juegos = juegos_ordenados.head(5)

    # Crear una lista de diccionarios con la información de los juegos top
    juegos_top = top_juegos[["title", "metascore"]].to_dict(orient="records")

    return juegos_top

# Función para obtener etiquetas de la columna deseada

@app.get("/etiquetas/")
def etiquetas(columna: str):
    if columna in df.columns:
        etiquetas = df[columna].explode().dropna().unique()
        return {columna: etiquetas.tolist()}
    else:
        return {"error": "No se tienen registros de ese dato."}


# Función
@app.get("/precio/")
def precio(publisher, genres, release_date, tags, specs, early_access, developer, sentiment):
    
    # Crear un DataFrame con los valores de entrada
    
    data = [[publisher, genres, release_date, tags, specs, early_access, developer, sentiment]]
    
    data = pd.DataFrame(data, columns=['publisher', 'genres','release_date', 'tags', 'specs','early_access', 'developer', 'sentiment'])
    
    #Lista de columnas a las que se aplicará  Encoding
    
    columnas = ["publisher","genres","release_date", "tags", "specs","sentiment","developer"]
    
    # Se aplica LabelEncoder a las columnas seleccionadas
    for col in columnas:
        data[col] = label_encoder.fit_transform(data[col])
    
       
    # Extraer el modelo del diccionario
    loaded_model = loaded_data['model']

    # Realizar la predicción utilizando el modelo cargado
    prediction = loaded_model.predict(data)
    prediction = prediction.flatten()

    # Crear un array con la misma longitud que loaded_y_train y llenarlo con la predicción
    prediction_array = [prediction] * len(loaded_y_train)

    # Calcular las métricas utilizando la predicción y el conjunto loaded_y_train
    mse = mean_squared_error(loaded_y_train, prediction_array)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(loaded_y_train, prediction_array)
    r2 = r2_score(loaded_y_train, prediction_array)

    # Combinar las métricas en una respuesta
    metrics = {
     'mse': mse,
     'rmse': rmse,
     'mae': mae,
     'r2': r2
    }

    response = {
        "prediction": prediction.tolist(),
        "metrics": metrics
    }

    return response