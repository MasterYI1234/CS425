import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

path1 = r"Test1_features.dat"
X = pd.read_csv(path1,engine ='python',header=None)
path2 = r"Test1_labels.dat"
Y = pd.read_csv(path2,engine ='python',header=None)

X.describe()

X = preprocessing.scale(X)

x = y = z = []
for C in range(1,10,1):
    for gamma in range(1,11,1):
        auc = cross_val_score(SVC(C=C,kernel='rbf',gamma=gamma/10),X,Y,cv=5,scoring='roc_auc').mean()
        x.append(C)
        y.append(gamma/10)
        z.append(auc)
x = np.array(x).reshape(9,10)
y = np.array(y).reshape(9,10)
z = np.array(z).reshape(9,10)


    # create a mesh to plot in
h=.02 # step size in the mesh
x_min, x_max = X[features[0]].min()-1, X[features[0]].max()+1
y_min, y_max = X[features[1]].min()-1, X[features[1]].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'Poly (C=10; D=2)',
          'Poly (C=100; D=2)',
          'Poly (C=10; D=3)',
          'RBF (C=10; G=0.1)',
          'RBF (C=100; G=0.1)',
          'RBF (C=10; G=0.5)',
          'RBF (C=100; G=0.5)']


# If we wanted to set a color scheme for our plot, we could do so here.
# For example:
#   pl.set_cmap(pl.cm.Accent)

for i, clf in enumerate((svc, poly_svc_one, poly_svc_two, poly_svc_three, rbf_svc_one, rbf_svc_two, rbf_svc_three, rbf_svc_four)):
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 4, i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #pl.set_cmap(pl.cm.Accent)
    
    # Apply 
    pl.contourf(xx, yy, Z)
    pl.axis('tight')

    # Plot also the training points
    pl.scatter(X[features[0]], X[features[1]], c=Y['target'], edgecolor='black')

    pl.title(titles[i])

pl.axis('tight')
pl.show()