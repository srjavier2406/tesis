import os
os.chdir(os.path.dirname(__file__))
import pandas as pd

# ========= Salida =========
ARCHIVO_SALIDA = "olist.csv"

# ========= Carga =========
ordenes   = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
items     = pd.read_csv("olist_order_items_dataset.csv")
productos = pd.read_csv("olist_products_dataset.csv")
reviews   = pd.read_csv("olist_order_reviews_dataset.csv")
cat_tr   = pd.read_csv("product_category_name_translation.csv") 

# ========= Categoría por producto =========
prod_cat = productos[["product_id", "product_category_name"]].merge(
    cat_tr, on="product_category_name", how="left"
)
prod_cat["categoria"] = prod_cat["product_category_name_english"]
prod_cat = prod_cat[["product_id", "categoria"]]

# ========= Fecha por orden =========
ordenes = ordenes[["order_id", "order_purchase_timestamp"]].rename(
    columns={"order_purchase_timestamp": "fecha_compra"}
)

# ========= Puntaje promedio por orden =========
rev = reviews[["order_id", "review_score"]].groupby("order_id", as_index=False)["review_score"].mean()

# ========= Base a nivel item =========
base = (
    items[["order_id", "order_item_id", "product_id", "price", "freight_value"]]
    .merge(prod_cat, on="product_id", how="left")
    .merge(ordenes, on="order_id", how="left")
    .merge(rev, on="order_id", how="left")
    .dropna(subset=["fecha_compra", "categoria"])
)

# ========= Semana  =========
base["semana"] = (
    base["fecha_compra"].dt.normalize()
    - pd.to_timedelta(base["fecha_compra"].dt.weekday, unit="D")
)

# ========= Agregación semanal por categoría =========
serie = (
    base.groupby(["categoria", "semana"], as_index=False)
    .agg(
        ventas=("order_item_id", "count"),
        precio_promedio=("price", "mean"),
        puntaje_promedio=("review_score", "mean"),
        costo_envio_promedio=("freight_value", "mean"),
    )
    .sort_values(["categoria", "semana"])
)

# ========= Regularizar semanas (crear semanas faltantes) =========
promedios = ["precio_promedio", "puntaje_promedio", "costo_envio_promedio"]

def regularizar(g):
    g = g.set_index("semana").sort_index()
    calendario = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
    g = g.reindex(calendario)
    g.index.name = "semana"

    g["categoria"] = g["categoria"].ffill().bfill()
    g["ventas"] = g["ventas"].fillna(0).astype(int)

    # Para evitar NaN en modelos: arrastrar últimos promedios válidos
    g[promedios] = g[promedios].ffill()

    return g.reset_index()
serie = serie.groupby("categoria", group_keys=False).apply(regularizar)

# ========= Dummies de eventos (semanas lunes) =========
cyber_monday = pd.to_datetime(["2016-11-28", "2017-11-27", "2018-11-26"])
navidad = pd.to_datetime(["2016-12-25", "2017-12-25"])
ano_nuevo = pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01"])

serie["navidad"] = serie["semana"].isin(navidad).astype(int)
serie["ano_nuevo"] = serie["semana"].isin(ano_nuevo).astype(int)
serie["cyber_monday"] = serie["semana"].isin(cyber_monday).astype(int)
serie.to_csv("olist.csv", index=False)
