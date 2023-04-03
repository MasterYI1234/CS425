from io import StringIO
from urllib import request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import numpy as np
import csv
import linearRegrassion as lg

cReader = pd.read_csv('auto-mpg.csv')

trainData = cReader[['mpg','displacement','acceleration']]
trainData.insert(0,'ones',1)

print(trainData.head(5))
cols = trainData.shape[1]
X = trainData.iloc[:,0:cols-1]
Y = trainData.iloc[:,cols-1:cols]
X = np.mat(X.values)
Y = np.mat(Y.values)

for i in range(1,3):
    X[:,i] = (X[:,i] - min(X[:,i])) / (max(X[:,i]) - min(X[:,i]))
print(X[:5:,:3])
#Y[:,0] = (Y[:,0] - min(Y[:,0])) / (max(Y[:,0]) - min(Y[:,0]))
print(Y[:5,0])

theta_n = (X.T*X).I*X.T*Y
print(theta_n)

theta = np.mat([0,0,0])
iters = 100000
alpha = 0.001

finalTheta,cost = lg.gradientDescent(X,Y,theta,alpha,iters)
print(finalTheta)
print(cost)

x1 = np.linspace(X[:,1].min(),X[:,1].max(),100)
x2 = np.linspace(X[:,2].min(),X[:,2].max(),100)


x1,x2 = np.meshgrid(x1,x2)
f = finalTheta[0,0] + finalTheta[0,1]*x1 + finalTheta[0,2]*x2

fig = plt.figure()
Ax = Axes3D(fig)
Ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap=cm.viridis,label='prediction')

Ax.scatter(X[:100,1],X[:100,2],Y[:100,0],c='y')
Ax.scatter(X[100:250,1],X[100:250,2],Y[100:250,0],c='r')
Ax.scatter(X[250:,1],X[250:,2],Y[250:,0],c='b')

Ax.set_zlabel('acceleration')  # 坐标轴
Ax.set_ylabel('displacement')
Ax.set_xlabel('mpg')

plt.show()

fig, bx = plt.subplots(figsize=(8,6))
bx.plot(np.arange(iters), cost, 'r') 
bx.set_xlabel('Iterations') 
bx.set_ylabel('Cost') 
bx.set_title('Error vs. Training Epoch') 
plt.show()