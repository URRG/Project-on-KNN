# pehla knowledge base system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics

iris=load_iris()
from sklearn.model_selection import train_test_split
X=iris.data[:]
y=iris.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2,random_state=4)
'''error1= []
error2= []
for k in range(1,60):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1= knn.predict(X_train)
    error1.append(np.mean(y_train!= y_pred1))
    y_pred2= knn.predict(X_test)
    error2.append(np.mean(y_test!= y_pred2))
# plt.figure(figsize(10,5))
plt.plot(range(1,60),error1,label="train")
plt.plot(range(1,60),error2,label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()'''

# training the knowledge base system
knn= KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)


classes={0:'setosa',1:'versicolor',2:'virginica'}

# taking values from user to predict

R = int(input("Enter the number of rows:"))
C = int(input("Enter the number of columns:"))
  
matrix = []
print("Enter the entries rowwise:")  
# input from user
for i in range(R):          
    a =[]
    for j in range(C):      
         a.append(float(input()))
    matrix.append(a)

y_pred=knn.predict(matrix)
print(classes[y_pred[0]])
print(classes[y_pred[1]])


