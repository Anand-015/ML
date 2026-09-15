import numpy as np
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

print("downloading fashion MNIST dataset(this may take 30-60 seconds)...")

fashion_mnist=fetch_openml('Fashion-MNIST',version=1,as_frame=False)

x=fashion_mnist.data
y=fashion_mnist.target.astype(int)

print(f"total dataset size:{x.shape[0]} images,each with {x.shape[1]} pixels.")

x_subset,_,y_subset,_=train_test_split(x,y,train_size=12000,stratify=y,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x_subset,y_subset,test_size=2000,stratify=y_subset,random_state=42)
print(f"Training image : {x_train.shape[0]}")
print(f"Testing image: {x_test.shape[0]}")

print(f"before scaling -> Min: {x_train.min()},Max: {x_train.max()}")
x_train=x_train/255.0
x_test=x_test/255.0
print(f"after scaling  -> Min: {x_train.min()},Max: {x_train.max()}")

k_values={1,3,5,7,9,15}
results={}
print(f"{'k Values':<8} |{'Accuarcy':<10} | {'prediction time (seconds)':<25}")
print("_"*50)

for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k,metric='euclidean',n_jobs=-1)
    knn.fit(x_train,y_train)
    start_time=time.time()
    y_pred=knn.predict(x_test)
    elapsed_time=time.time()-start_time
    acc=accuracy_score(y_test,y_pred)
    results[k]={
    "accuracy":acc,
    "time":elapsed_time,
    "predictions":y_pred
    }
    print(f"{k:<8} | {acc* 100:<9.2f}% | {elapsed_time:<25.2f}")


class_names={
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "sandal","shirt","sneaker","bag","ankle boot"
}
best_k=max(results,key=lambda k: results[k]["accuracy"])
print(f"Best K is : {best_k} with {results[best_k]['accuracy']*100:.2f} %accuracy\n")

print("pers-Class Classification Report :")
print(classification_report(y_test,results[best_k]["predictions"],target_names=class_names))

