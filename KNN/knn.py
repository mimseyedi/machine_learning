from math import dist, inf
from random import shuffle
from sklearn.datasets import load_iris as load
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNN:

    def __init__(self, k=5):
        self.__k = k
        self.__x_train = []
        self.__y_train = []

    def __euclid_distance(self, a, b):
        s = 0
        for ind in range(len(a)):
            s += (a[ind] - b[ind]) ** 2
        return s ** 0.5

    def __find_minimum(self, distances):
        min_val, min_ind = distances[0], 0
        for ind in range(1, len(distances)):
            if distances[ind] < min_val:
                min_val, min_ind = distances[ind], ind
        return min_ind

    def fit(self, x_train, y_train):
        self.__x_train = x_train
        self.__y_train = y_train

    def predict(self, x_test):
        predicts = list()
        for sample in x_test:
            distances = list()
            for row in self.__x_train:
                distances.append(self.__euclid_distance(sample, row))
            votes = dict()
            for _ in range(self.__k):
                min_ind = self.__find_minimum(distances)
                v = self.__y_train[min_ind]
                votes[v] = votes.get(v, 0) + 1
                distances[min_ind] = inf
            max_ind, max_val = 0, 0
            for key, value in votes.items():
                if value > max_val:
                    max_ind, max_val = key, value
            predicts.append(max_ind)
        return predicts

    def accuracy(self, y_test, predictions):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return correct / len(predictions)


x_train, x_test, y_train, y_test = train_test_split(load().data, load().target, test_size=0.2, shuffle=True)

model = KNN(k=5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("My KNN Accuracy Score:", model.accuracy(y_test, predictions))


# scikit-learn way->
model = KNeighborsClassifier(n_neighbors=5, algorithm='brute', weights='uniform')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("Scikit KNN Accuracy Score:", accuracy_score(y_test, predictions))