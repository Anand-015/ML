import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

housing =fetch_california_housing()
data=pd.DataFrame(housing.data,columns=housing.feature_names)
data['price']=housing.target

x=data[['AveRooms']].values
y=data['price'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_standard=scaler.fit_transform(x_train)
x_test_standard=scaler.fit_transform(x_test)

w=0
b=0
Lr=0.01
epochs=1000
cost_h=[]
n=len(x_train_standard)

for i in range(epochs):
    y_pred=w*x_train_standard.flatten()+b
    dw=(1/n)*np.sum((y_pred-y_train)*x_train_standard.flatten())
    db=(1/n)*np.sum(y_pred-y_train)
    w=w-Lr*dw
    b=b-Lr*db
    cost=(1/(2*n))*np.sum((y_pred-y_train)**2)
    cost_h.append(cost)
    if i%100==0:
       print(f"Epoch{i},cost={cost:.4f}")
y_pred_gd=w*x_test_standard.flatten()+b
print("gradient descent ")
print("w:",w)
print("b:",b)
print("MSE:",mean_squared_error(y_test,y_pred_gd))
print("R2_score:",r2_score(y_test,y_pred_gd))

x_mean=np.mean(x_train)
y_mean=np.mean(y_train)

w_ne=np.sum((x_train.flatten()-x_mean)*(y_train-y_mean))/np.sum((x_train.flatten()-x_mean)**2)
b_ne=y_mean-w_ne*x_mean


y_pred_ne=w_ne*x_test.flatten()+b_ne
print("Normal equation")
print("w",w_ne)
print("b",b_ne)
print("Mse",mean_squared_error(y_test,y_pred_ne))
print("R2",r2_score(y_test,y_pred_ne))


idx=np.argsort(x_test.flatten())
x_sorted=x_test.flatten()[idx]
y_sorted=y_test[idx]
y_pred_gd_sorted=y_pred_gd[idx]
y_pred_ne_sorted=y_pred_ne[idx]

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(cost_h,color="orange")
plt.title("Cost convergence")
plt.grid(True)

plt.subplot(1, 2, 2)

plt.scatter(x_test,y_test,color="blue",label="actaual")
plt.plot(x_sorted,y_pred_ne_sorted,color="red")
plt.xlabel("averoom")
plt.ylabel("price")
plt.title("linear regresssion")

plt.show()
