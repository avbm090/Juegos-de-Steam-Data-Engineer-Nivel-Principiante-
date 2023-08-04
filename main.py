import pandas as pd
import ast
from fastapi import FastAPI
import json


#rows=[]
#with open('steam_games.json', encoding='utf-8-sig') as f:
 #   for line in f.readlines():
  #    rows.append(ast.literal_eval(line))

#df=pd.DataFrame(rows)

app = FastAPI()
import fastparquet

# Lee el archivo Parquet y crea un objeto DataFrame
df = fastparquet.ParquetFile('steam_games.parquet').to_pandas()


# Función para obtener los géneros más vendidos
@app.get("/genero/")
def genero(Year: str):
    df_filtrado = df[df['year'] == Year]
    if df_filtrado.empty:
        return {"error": f"No hay datos disponibles {Year}"}
    
    generos_count = {}
    for lista_generos in df_filtrado['genres']:
        for genero_juego in lista_generos:
            generos_count[genero_juego] = generos_count.get(genero_juego, 0) + 1
    
    generos_top = dict(sorted(generos_count.items(), key=lambda item: item[1], reverse=True)[:5])
    return generos_top

# Función para obtener los juegos lanzados en un año
@app.get("/juegos/")
def juegos(anio: int):
   # Filtrar el DataFrame por el año deseado
    df_anio = df[df["release_date"].dt.year == anio]
    return df_anio["title"].unique()

# Función para obtener los specs más repetidos
@app.get("/specs/")
def specs(anio: int):
    df_filtrado = df[df["release_date"].dt.year == anio]
    if df_filtrado.empty:
        return {"error": f"No hay datos disponibles para el año {anio}"}
    
    specs_count = {}
    for lista_specs in df_filtrado['specs']:
        for spec in lista_specs:
            specs_count[spec] = specs_count.get(spec, 0) + 1
    
    top_specs = dict(sorted(specs_count.items(), key=lambda item: item[1], reverse=True)[:5])
    return {"top_specs": list(top_specs.keys())}

# Función para obtener la cantidad de juegos lanzados en un año con early access
@app.get("/earlyacces/")
def earlyacces(anio: int):
    df_anio=df[df["release_date"].dt.year==anio]
    frecuencias=df_anio["early_access"].sum()
    return frecuencias

# Función para obtener el análisis de sentimiento

@app.get("/sentiment/")
def sentiment(anio: int):
    df_anio = df[df["release_date"].dt.year==anio]
    frecuencia=df_anio["sentiment"].value_counts().to_dict()
    return frecuencia


# Función para obtener los juegos top 5 con mayor metascore
@app.get("/metascore/")
def metascore(anio: int):
    # Filtrar el DataFrame por el año proporcionado
    df_anio = df[df["release_date"].dt.year == anio]

    # Eliminar duplicados basados en la columna "title"
    juegos_unicos = df_anio.drop_duplicates(subset="title")

    # Ordenar los juegos únicos por Metascore de manera descendente
    juegos_ordenados = juegos_unicos.sort_values(by="metascore", ascending=False)

    # Tomar los 5 juegos top con los Metascores más altos
    top_juegos = juegos_ordenados.head(5)

    # Crear una lista de diccionarios con la información de los juegos top
    juegos_top = top_juegos[["title", "metascore"]].to_dict(orient="records")

    return juegos_top



# !uvicorn nombre_de_tu_archivo:app --reload --port 8001

#Después de ejecutar esta celda, tu API FastAPI se ejecutará en el puerto 8001 en tu entorno de Colab.
#Puedes acceder a la API en tu navegador web utilizando la URL: http://localhost:8001 o utilizar herramientas como
#requests en Python para realizar solicitudes a la API directamente desde el entorno de Colab.