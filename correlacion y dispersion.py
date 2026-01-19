
import os
os.chdir(os.path.dirname(__file__))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("olist.csv", parse_dates=["semana"])

# ==========================================
# 1. Seleccionar categoría
# ==========================================
CATEGORIA_OBJETIVO = "housewares"   # Cambiar por "health_beauty" o "housewares"
df_cat = df[df["categoria"] == CATEGORIA_OBJETIVO].copy()

# ==========================================
# 2. Definir variables a analizar
# ==========================================
variables_corr = [
    "ventas",
    "precio_promedio",
    "puntaje_promedio",
    "costo_envio_promedio",
    "navidad",
    "ano_nuevo",
    "cyber_monday"
]

# Filtrar solo columnas que existan realmente en el dataset
variables_corr = [col for col in variables_corr if col in df_cat.columns]

df_corr = df_cat[variables_corr]

# ==========================================
# 3. Matriz de correlación
# ==========================================
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Matriz de correlación — {CATEGORIA_OBJETIVO}")
plt.tight_layout()
plt.show()

# ==========================================
# 4. Gráficos de dispersión por variable numérica
# ==========================================

variables_dispersion = [
    "precio_promedio",
    "puntaje_promedio",
    "costo_envio_promedio",
]

for var in variables_dispersion:
    if var in df_cat.columns:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df_cat, x=var, y="ventas")
        plt.title(f"Ventas vs {var}, {CATEGORIA_OBJETIVO}")
        plt.tight_layout()
        plt.show()
        
