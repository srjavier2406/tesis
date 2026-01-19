import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np


def mostrar_tabla(nombre, archivo):
    print("\n" + "="*70)
    print(f"Tabla: {nombre}")
    df = pd.read_csv(archivo)

    print("\nPrimeras filas (head):")
    print(df.head())
    print("\nInformación general:")
    print(df.info())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    return df
tabla1 = mostrar_tabla("olist_customers_dataset.csv", "olist_customers_dataset.csv")
tabla2 = mostrar_tabla("olist_geolocation_dataset.csv", "olist_geolocation_dataset.csv")
tabla3 = mostrar_tabla("olist_order_items_dataset.csv", "olist_order_items_dataset.csv")
tabla4 = mostrar_tabla("olist_order_payments_dataset.csv", "olist_order_payments_dataset.csv")
tabla5 = mostrar_tabla("olist_order_reviews_dataset.csv", "olist_order_reviews_dataset.csv")
tabla6 = mostrar_tabla("olist_orders_dataset.csv", "olist_orders_dataset.csv")
tabla7 = mostrar_tabla("olist_products_dataset.csv", "olist_products_dataset.csv")
tabla8 = mostrar_tabla("olist_sellers_dataset.csv", "olist_sellers_dataset.csv")
tabla9 = mostrar_tabla("product_category_name_translation.csv", "product_category_name_translation.csv")

