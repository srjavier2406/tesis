import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

#--- Variables exógenas a considerar ---
exog_cols = [
    "precio_promedio",
    "puntaje_promedio",
    "costo_envio_promedio",
    "navidad",
    "ano_nuevo",
    "cyber_monday"
]

#--- agregar parámetros SARIMAX seleccionados en entrenamiento ---
order = (1, 1, 1)
seasonal_order = (0, 0, 0, 52) #dejar s en 52 por la anualidad

#--- Definir MAPE (se evita librería para manejar semanas sin ventas) ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

#--- Cargar y filtrar dataset ---
df = pd.read_csv("olist.csv")
df = df[df["categoria"] == "housewares"].copy()
#--- Formatear fecha semanal y ordenar cronológicamente ---
df["semana"] = pd.to_datetime(df["semana"])
df = df.sort_values("semana").set_index("semana")

#--- Quitar 4 filas para evaluar mismo horizonte con RF ---
df = df.iloc[4:].copy()

#--- Definir objetivo (y) y exógenas (X) ---
y = df["ventas"]
X = df[exog_cols]

#--- Partición temporal 80/20 en orden ---
train_size = int(len(df) * 0.8)

train_y = y.iloc[:train_size]
test_y  = y.iloc[train_size:]

train_X = X.iloc[:train_size]
test_X  = X.iloc[train_size:]

#--- Definir horizonte sobre el inicio del conjunto de prueba ---
test_y_h = test_y.iloc[:8]
test_X_h = test_X.iloc[:8]

#--- Ajuste SARIMAX (entrenamiento) ---
model = SARIMAX(
    train_y,
    exog=train_X,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

#--- Ajuste sobre train (visual) ---
fitted_train = results.fittedvalues

#--- Pronóstico 8 semanas (requiere exógenas futuras del horizonte) ---
forecast = results.get_forecast(steps=8, exog=test_X_h)
y_pred_h = forecast.predicted_mean
y_pred_h = pd.Series(y_pred_h.values, index=test_y_h.index)

#--- Métricas horizonte 4 semanas ---
y_true_4 = test_y_h.iloc[:4]
y_pred_4 = y_pred_h.iloc[:4]

mae_4  = mean_absolute_error(y_true_4, y_pred_4)
rmse_4 = np.sqrt(mean_squared_error(y_true_4, y_pred_4))
mape_4 = mean_absolute_percentage_error(y_true_4, y_pred_4)

#--- Métricas horizonte 8 semanas ---
y_true_8 = test_y_h.iloc[:8]
y_pred_8 = y_pred_h.iloc[:8]

mae_8  = mean_absolute_error(y_true_8, y_pred_8)
rmse_8 = np.sqrt(mean_squared_error(y_true_8, y_pred_8))
mape_8 = mean_absolute_percentage_error(y_true_8, y_pred_8)

#--- Totales reales vs pronosticados (4 y 8) ---
ventas_reales_4 = y_true_4.sum()
ventas_pred_4   = y_pred_4.sum()

ventas_reales_8 = y_true_8.sum()
ventas_pred_8   = y_pred_8.sum()

#--- Salida por consola ---
print("\nPronóstico SARIMAX")
print("HORIZONTE 4 SEMANAS")
print(f"  Ventas totales reales        : {ventas_reales_4:.2f}")
print(f"  Ventas totales pronosticadas : {ventas_pred_4:.2f}")
print(f"  MAE 4 semanas                : {mae_4:.2f}")
print(f"  RMSE 4 semanas               : {rmse_4:.2f}")
print(f"  MAPE 4 semanas               : {mape_4:.2f}%")

print("\nHORIZONTE 8 SEMANAS")
print(f"  Ventas totales reales        : {ventas_reales_8:.2f}")
print(f"  Ventas totales pronosticadas : {ventas_pred_8:.2f}")
print(f"  MAE 8 semanas                : {mae_8:.2f}")
print(f"  RMSE 8 semanas               : {rmse_8:.2f}")
print(f"  MAPE 8 semanas               : {mape_8:.2f}%")


# GRÁFICO

fin_4_sem = y_pred_h.index[3]
fin_8_sem = y_pred_h.index[7]

plt.figure(figsize=(14, 6))

# Serie real completa
plt.plot(df.index, df["ventas"], label="Ventas reales", linewidth=2)

# Ajuste en train 
warmup = 10
plt.plot(fitted_train.index[warmup:], fitted_train.iloc[warmup:], label="Ajuste SARIMAX (train)", linewidth=2)

# Pronóstico en test (8 semanas)
plt.plot(y_pred_h.index, y_pred_h, label="Pronóstico SARIMAX (test, 8 semanas)", linewidth=2)

# Corte train/test
plt.axvline(x=train_y.index[-1], color="gray", linestyle=":", label="Corte train/test")
plt.axvline(x=fin_4_sem, linestyle="--", label="Fin horizonte 4 semanas")
plt.axvline(x=fin_8_sem, linestyle="--", label="Fin horizonte 8 semanas")

plt.title(f"Ajuste y pronóstico SARIMAX")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
