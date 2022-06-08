import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
def yy(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))
def normalizar(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)
iris = pd.read_csv("Iris.csv")
iris.head()
g = sns.pairplot(iris.drop("Id", axis=1), hue="Species")
iris['Species'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2], inplace=True)
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x = pd.DataFrame(iris, columns=columns)
x = normalizar(x.to_numpy())
columns = ['Species']
y = pd.DataFrame(iris, columns=columns)
y = y.to_numpy()
y = y.flatten()
y = yy(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
w0 = 2*np.random.random((4, 5)) - 1
w1 = 2*np.random.random((5, 3)) - 1 
n = 0.1
errors = []
for i in range(1000):
    capa0 = X_train
    capa1 = sigmoid(np.dot(capa0, w0))
    capa2 = sigmoid(np.dot(capa1, w1))
    capa2_error = y_train - capa2
    capa2_delta = capa2_error * sigmoid_deriv(capa2)
    capa1_error = capa2_delta.dot(w1.T)
    capa1_delta = capa1_error * sigmoid_deriv(capa1)
    w1 += capa1.T.dot(capa2_delta) * n
    w0 += capa0.T.dot(capa1_delta) * n
    error = np.mean(np.abs(capa2_error))
    errors.append(error)
    accuracy = (1 - error) * 100     
print("Precision " + str(round(accuracy,2)) + "%")

