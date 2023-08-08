
from fastapi import FastAPI
import pandas as pd
import numpy as np
import fastparquet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Nombre del archivo Joblib

model_filename = 'gradient_boosting_model.joblib'

# Se carga el modelo, standardScaler y labelEncoder
with open(model_filename, 'rb') as file:
    model, scaler, encoder = joblib.load(file)

# Se carga el dataframe

df = pd.read_parquet('steam_games.parquet')

app = FastAPI()


# Función para obtener los géneros más vendidos
@app.get("/genero/")
def genero(anio: int): 
    df_filtrado = df[df['release_date'] == anio] # se filtra por el año seleccionado
    if df_filtrado.empty: 
        return {"!": f"No hay datos disponibles para el año {anio}"}
    # se utiliza un for para poder correr más rápido los datos,
    # en este caso se están desanidando manualmente para no consumir mucha ram en render
    contador_generos = {}
    for lista_generos in df_filtrado['genres']:
        if lista_generos is not None: # se evitan los valores nulos
            for genero_juego in lista_generos:
                if genero_juego is not None: # si el valor no es nulo se comienza a almacenar en el contador
                    contador_generos[genero_juego] = contador_generos.get(genero_juego, 0) + 1
    
    generos_top = dict(sorted(contador_generos.items(), key=lambda item: item[1], reverse=True)[:5])

    return generos_top


# Función para obtener los juegos lanzados en un año

@app.get("/juegos/")
def juegos(anio: int):
    
   # Se filtra el DataFrame por el año deseado
   
    df_filtrado = df[df['release_date'] == anio]
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    juegos_unicos = df_filtrado["title"].unique()
    
    respuesta = {"juegos": juegos_unicos.tolist()}  # se genera un diccionario con los elementos de la lista 
    return respuesta

# Función para obtener los specs más repetidos
@app.get("/specs/")
def specs(anio: int):
    df_filtrado = df[df['release_date'] == anio] # se filtra por año
    if df_filtrado.empty: # se corrobora que no hayan nulos, en tal caso se muestra el mensaje que precede
        return {"!": f"No hay datos disponibles para el año {anio}"}
    
    specs_count = {} # se genera un contador
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
    # Se filtra el dataframe por año
    df_filtrado = df[df['release_date'] == anio]
    
    # Se muestra un mensaje, en caso de que haya valores nans
    if df_filtrado.empty:
        return {"!": f"No hay datos disponibles para el año {anio}"}
    
    # Se cuentan la cantidad de datos y se genera una lista
    frecuencia=df_filtrado["sentiment"].value_counts().to_dict()
    
    return frecuencia

# Función para obtener los juegos top 5 con mayor metascore
@app.get("/metascore/")
def metascore(anio: int):
    # Se filtra el DataFrame por el año proporcionado
    df_anio = df[df["release_date"]== anio]

    # Se eliminan duplicados basados en la columna "title"
    juegos_unicos = df_anio.drop_duplicates(subset="title")

    # Se ordenan los juegos únicos por Metascore de manera descendente
    juegos_ordenados = juegos_unicos.sort_values(by="metascore", ascending=False)

    # Se toman los 5 juegos top con los Metascores más altos
    top_juegos = juegos_ordenados.head(5)

    # Se crea una lista de diccionarios con la información de los juegos top
    juegos_top = top_juegos[["title", "metascore"]].to_dict(orient="records")

    return juegos_top

# Función para obtener etiquetas de la columna deseada

@app.get("/etiquetas/")
def etiquetas(columna: str):
    
    # se desanidan los datos con esta función, 
    # esto se hace porque se corroboró que no hay inconvenientes
    # con la ram, se dejan las funciones anteriores con los códigos originales, 
    # pero se podrían haber desanidado de la misma manera corroborado esto.
    
    if columna in df.columns:
        etiquetas = df[columna].explode().dropna().unique()  
        return {columna: etiquetas.tolist()}
    else:
        return {"error": "No se tienen registros de ese dato."}


# Función
@app.get("/precio/")
def precio(publisher, genres, release_date, tags, specs, early_access, developer, sentiment):
    
    # Se crea un DataFrame con los valores de entrada
    
    data = [[publisher, genres, release_date, tags, specs, early_access, developer, sentiment]]
    
    data = pd.DataFrame(data, columns=['publisher', 'genres','release_date', 'tags', 'specs','early_access', 'developer', 'sentiment'])
    
    # Se genera una lista de columnas a las que se aplicará  Encoding
    
    columnas = ["publisher","genres","release_date", "tags", "specs","sentiment","developer"]
    
    # Se aplica LabelEncoder a las columnas seleccionadas
    for col in columnas:
        data[col] = encoder.fit_transform(data[col])


    # Se realiza la predicción utilizando el modelo cargado
    prediction = model.predict(data)
    prediction = prediction.flatten()

    response = {
        "predicción_precio": prediction.tolist(),
        "metricas": {
            "MSE": 21.313749278012445,
            "RMSE: ": 4.616681630566748,
            "MAE": 3.560192284621895,
            "R2": 0.2938208845136231
        }
    }
    
    return response