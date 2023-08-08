import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastparquet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

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

# Se crea un objeto LabelEncoder, este se guardará para su posterior uso
label_encoder = LabelEncoder()

# Se aplica LabelEncoder a las columnas seleccionadas
for col in columnas:
    X[col] = label_encoder.fit_transform(X[col])

# Se estandarizan los datos utilizando StandardScaler, este objeto se guardará para su posterior uso
scaler = StandardScaler()
X_n = scaler.fit_transform(X)

# Se convierte el arreglo numpy a un DataFrame de pandas, manteniendo el nombre de las columnas
X = pd.DataFrame(data=X_n, columns=X.columns)

# Se dividen los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Se entrena el modelo el modelo
model = GradientBoostingRegressor(learning_rate=0.2, max_depth=5, n_estimators=200, random_state=42)  
model.fit(X_train, y_train)

# Se realiza la predicción en el conjunto de prueba
prediction = model.predict(X_test)

# Cálculo de métricas
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

# Nombre del archivo Joblib
model_filename = 'gradient_boosting_model.joblib'

# Se guarda el modelo, StandardScaler y LabelEncoder en formato Joblib
with open(model_filename, 'wb') as file:
    joblib.dump((model, scaler, label_encoder), file)


# Se muestran las métricas
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)




