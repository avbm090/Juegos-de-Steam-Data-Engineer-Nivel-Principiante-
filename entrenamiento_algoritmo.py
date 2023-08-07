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


df = fastparquet.ParquetFile('steam_games.parquet').to_pandas()

# Se desanidan las listas en las columnas "genres" y "specs"
df["genres"] = df["genres"].explode().reset_index(drop=True)
df["specs"] = df["specs"].explode().reset_index(drop=True)
df["tags"] = df["tags"].explode().reset_index(drop=True)

# Se filtran las filas donde la fecha sea desde el año 2010 en adelante
df= df.loc[df['release_date'] >= 2010]

df = df.dropna(subset=["price"])

df = df[df["price"] <= 22]

df.drop(columns=["metascore"],inplace=True) # se elimina dicha columna

df.dropna(subset=["publisher"],inplace=True)

df.dropna(subset=["genres"],inplace=True)

df.dropna(subset=["developer"],inplace=True)

df.dropna(subset=["publisher"],inplace=True)

df.dropna(subset=["genres"],inplace=True)

df.dropna(subset=["specs"],inplace=True)

df.dropna(subset=["tags"],inplace=True)

X=df.drop("price",axis=1)
y=df['price']

X=X.drop("title",axis=1)

#Lista de columnas a las que se aplicará  Encoding
columnas = ["publisher","genres", "tags", "specs","sentiment","developer","release_date"]

# Se crea un objeto LabelEncoder
label_encoder = LabelEncoder()

# Se aplica LabelEncoder a las columnas seleccionadas
for col in columnas:
    X[col] = label_encoder.fit_transform(X[col])

# Se estandarizan los datos utilizando StandardScaler
scaler = StandardScaler()
X_n = scaler.fit_transform(X)

# Se convierte el arreglo numpy a un DataFrame de pandas, manteniendo el nombre de las columnas
X = pd.DataFrame(data=X_n, columns=X.columns)


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Se entrena el modelo
model = GradientBoostingRegressor(learning_rate=0.2, max_depth=5, n_estimators=200)
model.fit(X_train, y_train)

# Guardar el modelo y y_train en formato pickle
model_and_data = {
    'model': model,
    'y_train': y_train
}

model_filename = 'gradient_boosting_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model_and_data, file)




