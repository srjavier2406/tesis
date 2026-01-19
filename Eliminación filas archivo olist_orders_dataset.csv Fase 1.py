import pandas as pd
import numpy as np
tabla6 = pd.read_csv('olist_orders_dataset.csv')
print("Tabla: olist_orders_dataset.csv")
print(tabla6.head())
print(tabla6.info())
print(tabla6.isnull().sum())
tabla6 = tabla6.dropna(subset=['order_delivered_customer_date'])
tabla6 = tabla6.dropna(subset=['order_approved_at'])
tabla6 = tabla6.dropna(subset=['order_delivered_carrier_date'])
print(tabla6.isnull().sum())
tabla6.to_csv('olist_orders_dataset.csv', index=False)
