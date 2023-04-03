import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)


    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()


    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()

    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)

    print(classification_report(y_true, y_pred))

    print()


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