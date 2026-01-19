import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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

#--- Modelo RF (ajustar a hiperparámetros seleccionados en entrenamiento) ---
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

#--- Definir MAPE (se evita librería para manejar semanas sin ventas) ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

#--- Pronóstico recursivo RF multi-pasos ---
def rf_forecast_recursive(model, y_hist, exog_future):

    #--- Mantener histórico de tamaño 4 (para lag_4) ---
    hist = list(pd.Series(y_hist).iloc[-4:].astype(float).values)
    preds = []

    for i in range(len(exog_future)):
        feats = exog_future.iloc[i].to_dict()
        feats["lag_1"] = hist[-1]
        feats["lag_2"] = hist[-2]
        feats["lag_4"] = hist[-4]

        X_row = pd.DataFrame([feats], columns=exog_cols + ["lag_1", "lag_2", "lag_4"])

        #--- Predicción y actualización recursivo ---
        y_hat = float(model.predict(X_row)[0])
        preds.append(y_hat)

        hist.append(y_hat)
        hist = hist[-4:]

    #--- Devolver predicciones con el mismo índice temporal ---
    return pd.Series(preds, index=exog_future.index)

#--- Cargar y filtrar dataset ---
df = pd.read_csv("olist.csv")
df = df[df["categoria"] == "housewares"].copy()

#--- Formatear fecha semanal y ordenar cronológicamente ---
df["semana"] = pd.to_datetime(df["semana"])
df = df.sort_values("semana").set_index("semana")

#--- Crear rezagos (lags) para incorporar dependencia temporal ---
df["lag_1"] = df["ventas"].shift(1)
df["lag_2"] = df["ventas"].shift(2)
df["lag_4"] = df["ventas"].shift(4)

#--- Eliminar filas iniciales con NaN por creación de lags ---
df = df.dropna(subset=["lag_1", "lag_2", "lag_4"]).copy()

#--- Definir columnas de entrada (exógenas + lags) ---
feature_cols = exog_cols + ["lag_1", "lag_2", "lag_4"]

#--- Partición temporal 80/20 (sin mezclar el orden de la serie) ---
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test  = df.iloc[train_size:]

#--- Variable objetivo (y) ---
y_train = train["ventas"]
y_test  = test["ventas"]

#--- Variables de entrada (X) ---
X_train = train[feature_cols]
X_test  = test[feature_cols]

#--- Definir horizonte sobre el inicio del conjunto de prueba ---
y_test_h = y_test.iloc[:8]
X_test_h = X_test.iloc[:8]

#--- Entrenar modelo RF ---
rf_model.fit(X_train, y_train)

#--- Ajuste en entrenamiento (visualización) ---
train_pred = pd.Series(rf_model.predict(X_train), index=train.index)

#--- Preparar exógenas futuras (primeras 8 semanas del test) ---
X_exog_future = test[exog_cols].iloc[:8].copy()

#--- Pronóstico recursivo 8 semanas ---
y_pred_h = rf_forecast_recursive(rf_model, y_train, X_exog_future)

#--- Métricas horizonte 4 semanas ---
y_true_4 = y_test_h.iloc[:4]
y_pred_4 = y_pred_h.iloc[:4]

mae_4  = mean_absolute_error(y_true_4, y_pred_4)
rmse_4 = np.sqrt(mean_squared_error(y_true_4, y_pred_4))
mape_4 = mean_absolute_percentage_error(y_true_4, y_pred_4)

#--- Métricas horizonte 8 semanas ---
y_true_8 = y_test_h.iloc[:8]
y_pred_8 = y_pred_h.iloc[:8]

mae_8  = mean_absolute_error(y_true_8, y_pred_8)
rmse_8 = np.sqrt(mean_squared_error(y_true_8, y_pred_8))
mape_8 = mean_absolute_percentage_error(y_true_8, y_pred_8)

#--- Totales reales vs pronosticados (4 y 8) ---
ventas_reales_4 = y_true_4.sum()
ventas_pred_4   = y_pred_4.sum()

ventas_reales_8 = y_true_8.sum()
ventas_pred_8   = y_pred_8.sum()


print("\nPronóstico Random Forest")
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


# =========================
# GRÁFICO
# =========================
fin_4_sem = y_pred_h.index[3]
fin_8_sem = y_pred_h.index[7]

plt.figure(figsize=(14, 6))

plt.plot(df.index, df["ventas"], label="Ventas reales", linewidth=2)
plt.plot(train_pred.index, train_pred, label="Ajuste Random Forest (train)", linewidth=2)
plt.plot(y_pred_h.index, y_pred_h, label="Pronóstico Random Forest (test, 8 semanas)", linewidth=2)

plt.axvline(x=train.index[-1], color="gray", linestyle=":", label="Corte train/test")
plt.axvline(x=fin_4_sem, linestyle="--", label="Fin horizonte 4 semanas")
plt.axvline(x=fin_8_sem, linestyle="--", label="Fin horizonte 8 semanas")

plt.title(f"Pronóstico Random Forest")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
