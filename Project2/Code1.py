import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import math
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from numpy import dot



#Standardization
scaler = StandardScaler().fit('auto-mpg.csv'))
standardizedX = scaler.transform('auto-mpg.csv')

cars = pd.read_csv('auto-mpg.csv')


#First draw to confirm
plt.scatter(cars["mpg"] ,cars["acceleration"]) 
plt.xlabel('mpg')
plt.ylabel('acceleration')
plt.show()

data = cars[['mpg','acceleration']]
data.insert(0, 'Ones', 1) #Insert a column number of all 1s between columns 1-2 of data
# set X (training data) and y (target variable)
cols = data.shape[1]  #Calculate the number of data columns, where cols is 3
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]


X = np.matrix(X.values)  
y = np.matrix(y.values)  #Convert variables from data frames to matrix form
theta_n = dot(dot(inv(dot(X.T, X)), X.T), y) # theta = (X'X)^(-1)X'Y
print theta_n

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
return np.sum(inner) / (2 * len(X))
X.shape, theta_n.shape, y.shape
lr_cost = computeCost(X, y, theta_n.T)
print(lr_cost)