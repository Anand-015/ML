import pandas as pd
import numpy as np
df = pd.read_excel("Online Retail.xlsx")
print("Original records:",len(df))
df=df.dropna(subset=["CustomerID"])
df=df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df=df[(df["Quantity"]>0)&(df["UnitPrice"]>0)]
df["TotalAmount"]=df["Quantity"]*df["UnitPrice"]
customer=df.groupby("CustomerID").agg(
    Frequency=("InvoiceNo","nunique"),
    TotalQuantity=("Quantity","sum"),
    UniqueProducts=("StockCode","nunique"),
    Monetary=("TotalAmount","sum")).reset_index()

customer["AvgBasketValue"]=(
    customer["Monetary"]/customer["Frequency"])
customer["Segment"] = pd.qcut(
    customer["Monetary"],
    q=3,
    labels=["Low","Medium","High"]
)
print("\n Customer Segments")
print(customer["Segment"].value_counts())
features=[
    "Frequency",
    "TotalQuantity",
    "UniqueProducts",
    "Monetary",
    "AvgBasketValue"]

for feature in features:
    customer[feature] = pd.cut(
        customer[feature],
        bins=3,
        labels=["Low","Medium","High"],
        include_lowest=True
    )
  def entropy(data):
    counts=data["Segment"].value_counts()
    total=len(data)
    ent=0
    for count in counts:
        p=count/total
        if p>0:
            ent -=p*np.log2(p)
            return ent
total_entropy=entropy(customer)
print("\nOverall Entropy=",round(total_entropy,4))
