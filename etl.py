import pandas as pd
import ast
from fastapi import FastAPI
import json
import fastparquet

# Para poder leer los datos correctamente, se codifican primero
# por filas, luego se guardan como formato dataframe.
rows=[]
with open('steam_games.json', encoding='utf-8-sig') as f:
    for line in f.readlines():
      rows.append(ast.literal_eval(line))

df=pd.DataFrame(rows)

# Se transforma la columna "release_date" a otra nueva que sólo muestre el año
df["release_date"] = pd.to_datetime(df["release_date"], format='%Y-%m-%d', errors='coerce').dt.year


# Se cambia el formato de la columna "early_access" de tipo booleano a valor numérico
df["early_access"]=df["early_access"].astype(int)

# Se reemplazan los valores tipo "nan" por la clasificación "sin_calificar"
df["sentiment"].fillna("sin_calificar",inplace=True)

# Se convierten los valores de la columna metascore a numéricos
df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")

# Se convierte la columna 'precio' a formato numérico
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Se eliminan las columnas que no serán necesarias para las consultas

df.drop(['app_name','url','discount_price','reviews_url','id'], axis=1, inplace=True)

# Se guarda el DataFrame en formato parquet
df.to_parquet('steam_games.parquet', index=False)



