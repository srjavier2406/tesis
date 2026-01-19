import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("olist.csv", parse_dates=["semana"])

# Crear tabla 
resumen = []

# Por categoría
for cat in df["categoria"].dropna().unique():
    data_cat = df[df["categoria"] == cat].sort_values("semana")

    # Variables base
    ventas = data_cat["ventas"].values
    semanas = np.arange(len(data_cat)).reshape(-1, 1)

    # Volatilidad
    volatilidad = np.std(ventas) / np.mean(ventas)

    # Tendencia 
    modelo_lin = LinearRegression()
    modelo_lin.fit(semanas, ventas)
    tendencia = modelo_lin.coef_[0]

    # Estacionalidad
    estacionalidad = pd.Series(ventas).autocorr(lag=52)
   
    # Ventas total / categorias
    total_ventas = data_cat["ventas"].sum()

    resumen.append({
        "categoria": cat,
        "volatilidad": volatilidad,
        "tendencia": tendencia,
        "estacionalidad": estacionalidad,
        "total_ventas": total_ventas
    })

# Convertir a DataFrame
resumen_df = pd.DataFrame(resumen)

# Comparación
for col in ["volatilidad", "tendencia", "estacionalidad"]:
    resumen_df[col + "_abs"] = resumen_df[col].abs()
print("\n Categorías más volátiles:")
print(resumen_df.sort_values("volatilidad_abs", ascending=False).head(40)[["categoria","volatilidad","total_ventas"]])
print("\n Categorías con mayor tendencia:")
print(resumen_df.sort_values("tendencia_abs", ascending=False).head(15)[["categoria","tendencia","total_ventas"]])
print("\n Categorías con mayor estacionalidad:")
print(resumen_df.sort_values("estacionalidad_abs", ascending=False).head(15)[["categoria","estacionalidad","total_ventas"]])


