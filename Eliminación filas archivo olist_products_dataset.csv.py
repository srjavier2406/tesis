import pandas as pd
import numpy as np
tabla7 = pd.read_csv('olist_products_dataset.csv')
print("Tabla: olist_products_dataset.csv")
print(tabla7.head())
print(tabla7.info())
print(tabla7.isnull().sum())
tabla7 = tabla7.dropna(subset=['product_category_name'])
print(tabla7.isnull().sum())
tabla7.to_csv('olist_products_dataset.csv', index=False)