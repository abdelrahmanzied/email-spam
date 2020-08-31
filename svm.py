#Import
import numpy as np
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

#Data
data_train = loadmat('spamTrain.mat')
data_test = loadmat('spamTest.mat')

X_train = data_train['X']
X_test = data_test['Xtest']
y_train = data_train['y'].ravel()
y_test = data_test['ytest'].ravel()

#Model
svc = svm.SVC()
svc.fit(X_train, y_train)

# Score
print('Train accuracy = {0}%'.format(np.round(svc.score(X_train, y_train) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(X_test, y_test) * 100, 2)))

#Predict
y_pred = svc.predict(X_test)

#Accuracy Score
AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score: = {0}%'.format(np.round(AccScore * 100, 2)))

#Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', CM)
sns.heatmap(CM, center=True)