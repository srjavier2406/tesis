import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np

print("Tabla: olist.csv ")
olist = pd.read_csv("olist.csv")
print("\nPrimeras filas (head):")
print(olist.head(30))
print("\nInformación general:")
print(olist.info())
print("\nValores nulos por columna:")
print(olist.isnull().sum())

