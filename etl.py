import pandas as pd
import ast
from fastapi import FastAPI
import json
import fastparquet

rows=[]
with open('steam_games.json', encoding='utf-8-sig') as f:
    for line in f.readlines():
      rows.append(ast.literal_eval(line))

df=pd.DataFrame(rows)

df["release_date"] = pd.to_datetime(df["release_date"],errors='coerce')

# se desanidan las listas en las columnas 'genres', 'tags', 'specs', etc.
df = df.explode('genres').explode('tags').explode('specs')

# Se cambia el formato de la columna "early_access" de tipo booleano a valor numérico
df["early_access"]=df["early_access"].astype(int)

# Se reemplazan los valores tipo "nan" por la clasificación "sin_calificar"
df["sentiment"].fillna("sin_calificar",inplace=True)

#Convertimos los valores de la columna metascore a numéricos
df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")

# Se convierte la columna 'precio' a formato numérico
df['price'] = pd.to_numeric(df['price'], errors='coerce')

#eliminamos las columnas que no serán necesarias para las consultas

df.drop(['publisher', 'app_name','url','tags','discount_price','reviews_url','price','id','developer'], axis=1, inplace=True)

# Guarda el DataFrame en formato Parquet
df.to_parquet('steam_games.parquet', index=False)



