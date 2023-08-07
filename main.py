import pandas as pd
from fastapi import FastAPI
import json
import fastparquet
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import LabelEncoder


# Se crea un objeto LabelEncoder
label_encoder = LabelEncoder()

# Se carga el modelo Boosting con sus métricas

with open("gradient_boosting_model_con_metricas.pkl", "rb") as file:
    loaded_data = pickle.load(file)

modelo_boosting = loaded_data['model']
loaded_mse = loaded_data['mean_squared_error']
loaded_rmse = loaded_data['root_mean_squared_error']
loaded_mae = loaded_data['mean_absolute_error']
loaded_r2 = loaded_data['r2_score'] 

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

    # Se realiza la predicción utilizando el modelo cargado
    prediction = modelo_boosting.predict(data)
    
    # Métricas
    
    print("Métricas del modelo:")
    print("Mean Squared Error:", loaded_mse)
    print("Root Mean Squared Error:", loaded_rmse)
    print("Mean Absolute Error:", loaded_mae)
    print("R2 :", loaded_r2)
   
    return {"prediction": prediction[0]}
