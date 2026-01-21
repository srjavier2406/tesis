import os
os.chdir(os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------Se define MAPE (no por librería para evitar inconsistencia por semanas sin ventas)---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

#------Preparación de serie para modelos --------------------------------------------------
df = pd.read_csv("olist.csv")
df = df[df["categoria"] == "housewares"].copy()
df["semana"] = pd.to_datetime(df["semana"])
#-------ordenar cronológico-------
df = df.sort_values("semana").set_index("semana")
#------Variables externas a considerar--------
exog_cols = [
    "precio_promedio",
    "puntaje_promedio",
    "costo_envio_promedio",
    "navidad",
    "ano_nuevo",
    "cyber_monday"
]
#-- eliminar 4 primeras semanas para igualar horizonte con RF (por rezagos)------------------
df = df.iloc[4:].copy()
#-------------Partición 80/20------------------------------
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
#---- variable a predecir---
y_train = train["ventas"]
#---- variables de entrada----
X_train = train[exog_cols]
#--- rango de parámetros (lo mismo que rf )
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

seasonal_pdq = [
    (P, D, Q, 52)
    for P, D, Q in itertools.product(range(0, 2), range(0, 2), range(0, 2))
]
results_list = []
best_mape = np.inf
best_params = None
best_model = None
best_rmse = None
best_mae = None
#---------- busqueda de combinaciones exahustiva
print("buscando parámetros\n")

for order in pdq:
    for seasonal_order in seasonal_pdq:
        try:
            print(f"Probando SARIMAX{order}x{seasonal_order}...", end=" ")

            model = SARIMAX(
                y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            fitted = results.fittedvalues
            y_aligned = y_train.loc[fitted.index]

            mae  = mean_absolute_error(y_aligned, fitted)
            rmse = np.sqrt(mean_squared_error(y_aligned, fitted))
            mape = mean_absolute_percentage_error(y_aligned, fitted)
            results_list.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                
            })

            if not np.isnan(mape) and mape < best_mape: #se guarda mejor parametro bajo critero MAPE
                best_mape = mape
                best_params = (order, seasonal_order)
                best_model = results
                best_mae = mae
                best_rmse = rmse

            print(f"MAPE={mape:.2f}%, MAE={mae:.2f}, RMSE={rmse:.2f}")

        except Exception as e:
            print(f"falló ({e})")
            continue


print("\nMejores parámetros (según menor MAPE en entrenamiento):")
print(f"order          = {best_params[0]}")
print(f"seasonal_order = {best_params[1]}")
print(f"MAPE           = {best_mape:.2f}%")
print(f"MAE            = {best_mae:.2f}")
print(f"RMSE           = {best_rmse:.2f}")

fitted_train = best_model.fittedvalues
print(best_model.summary())
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["ventas"], label="Actual sales", linewidth=2)
plt.plot(fitted_train.index, fitted_train,
         label="Ajuste SARIMAX (train)", linewidth=2)

split_date = train.index[-1]
plt.axvline(split_date, color="gray", linestyle=":", label="Train/Test split")

plt.title("Mejor modelo SARIMAX (según menor MAPE en entrenamiento)")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

