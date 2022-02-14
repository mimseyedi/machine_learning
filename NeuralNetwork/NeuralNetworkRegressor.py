from numpy import append, ones
from numpy.random import rand
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston as boston
from sklearn.neural_network import MLPRegressor


class NeuralNetworkRegressor:
    def __init__(self, learning_rate=0.01, max_iter=150):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def fit(self, x_train, y_train):
        rows, cols = x_train.shape
        weights = rand(cols)
        for t in range(self.max_iter):
            for i in range(rows):
                y_p = x_train[i] @ weights
                error = y_p - y_train[i]
                weights -= self.learning_rate * error * x_train[i]
        self.weights = weights

    def predict(self, x_test):
        return x_test @ self.weights


data = boston().data
labels = boston().target
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = append(ones(shape=(len(x_train), 1)), x_train, axis=1)
x_test = append(ones(shape=(len(x_test), 1)), x_test, axis=1)

nn_model = NeuralNetworkRegressor(learning_rate=0.01, max_iter=200)
nn_model.fit(x_train, y_train)
predictions = nn_model.predict(x_test)
print('My NNR Model MAE:', mean_absolute_error(y_test, predictions))

# scikit-learn way:
sk_model = MLPRegressor(alpha=0.01, max_iter=200)
sk_model.fit(x_train, y_train)
predictions = sk_model.predict(x_test)
print('Scikit NNR Model MAE:', mean_absolute_error(y_test, predictions))
