import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------Se define MAPE (no por librería para evitar inconsistencia por semanas sin ventas)------------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#------Preparación de serie para modelos --------------------------------------------------
df = pd.read_csv("olist.csv")
df = df[df["categoria"] == "health_beauty"].copy() #aquí cambiar la categoria (housewares, electronics, health_beauty)
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
    "cyber_monday"]

#------ Crear dependencia RF------------
df["lag_1"] = df["ventas"].shift(1)
df["lag_2"] = df["ventas"].shift(2)
df["lag_4"] = df["ventas"].shift(4)

df = df.dropna(subset=["lag_1", "lag_2", "lag_4"]).copy()
feature_cols = exog_cols + ["lag_1", "lag_2", "lag_4"]
#-------------Partición 80/20------------------------------
n_total   = len(df)
train_len = int(n_total * 0.8)
train = df.iloc[:train_len]
test  = df.iloc[train_len:]
#---- variable a predecir---
y_train = train["ventas"]
#---- variables de entrada----
X_train = train[feature_cols]


# -------Hiperparámetros (Grid Search con for)------------------------------
n_estimators_list    = [10, 50,100, 200, 300]
max_depth_list       = [None, 5, 10]
min_samples_split_ls = [2, 5]
min_samples_leaf_ls  = [1, 2]

best_mape = np.inf
best_params = None
best_model = None
best_mae = None
best_rmse = None
print("búsqueda de hiperparámetros\n")
for n_est in n_estimators_list:
    for max_d in max_depth_list:
        for min_split in min_samples_split_ls:
            for min_leaf in min_samples_leaf_ls:
                params = {
                    "n_estimators": n_est,
                    "max_depth": max_d,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf,
                    "random_state": 42,
                    "n_jobs": -1
                }
                try:
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)

                    y_pred_train = model.predict(X_train)

                    mae_train  = mean_absolute_error(y_train, y_pred_train)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    mape_train = mape(y_train, y_pred_train)

                    if not np.isnan(mape_train) and mape_train < best_mape: #guardar mejor hiperparámetro segun menor MAPE
                        best_mape = mape_train
                        best_params = params
                        best_model = model
                        best_mae = mae_train
                        best_rmse = rmse_train

                    print(f"RF {params}  MAPE={mape_train:.2f}%")
                    
                except Exception as e:
                    print(f"falló {params} ({e})")
                    continue

print(best_params)
print(f"MAE  : {best_mae:.2f}")
print(f"RMSE : {best_rmse:.2f}")
print(f"MAPE : {best_mape:.2f}%\n")
#------------------------------------------------------------
y_pred_train = best_model.predict(X_train)
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["ventas"], label="Ventas reales", linewidth=2)
plt.plot(train.index, y_pred_train,
             label="Ajuste RF (train)",
             linewidth=2, linestyle="-")

split_date = train.index[-1]
plt.axvline(split_date, color="gray", linestyle=":", label="Separación train/test")

plt.title("Mejor Random Forest (según menor MAPE en entrenamiento)")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()