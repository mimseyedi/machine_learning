import numpy as np
from numpy.core.numeric import ones
from numpy.lib.function_base import append
from numpy.linalg import inv
from sklearn.datasets import load_boston as load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class LeastSquares:

    def __init__(self, fit_intercept=True):
        self.__fit_intercept = fit_intercept
        self.__weights = []

    def __add_intercept(self, data):
        intercept = ones((len(data), 1))
        return append(intercept, data, axis=1)

    def fit(self, x_train, y_train):
        if self.__fit_intercept:
            X = self.__add_intercept(x_train)
            self.__weights = inv(X.T @ X) @ X.T @ y_train
        else:
            self.__weights = inv(x_train.T @ x_train) @ x_train.T @ y_train

    def predict(self, x_test):
        if self.__fit_intercept:
            X = self.__add_intercept(x_test)
            predictions = X @ self.__weights
            return predictions
        return x_test @ self.__weights

    def mse(self, y_test, predictions):
        return ((y_test - predictions) ** 2).mean()

    def mae(self, y_test, predictions):
        return abs(y_test - predictions).mean()


x_train, x_test, y_train, y_test = train_test_split(scale(load().data), load().target, test_size=0.1, shuffle=True)

model = LeastSquares()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("My LinearRegression MAE:", model.mae(y_test, predictions))

# scikit-learn way->
reg = LinearRegression(fit_intercept=True, normalize=False)
reg.fit(x_train, y_train)
predictions = reg.predict(x_test)
print("Scikit LinearRegression MAE:", mean_absolute_error(y_test, predictions))
