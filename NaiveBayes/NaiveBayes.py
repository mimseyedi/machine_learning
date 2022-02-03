import numpy as np
from statistics import mean, pvariance
from math import pi, exp
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris as iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class NaiveBayes:

    def __init__(self):
        self.__ft_dict = dict()
        self.__yi_dict = dict()

    def __gauss_probability(self, x_vec):
        rs_dict = {}
        for f_key in self.__ft_dict:
            result = self.__yi_dict[f_key] / len(y_train)
            for s_key in range(len(x_vec)):
                avg = self.__ft_dict[f_key][s_key]['avg']
                var = self.__ft_dict[f_key][s_key]['var']
                step1 = exp(-((x_vec[s_key] - avg) ** 2) / (2 * var))
                step2 = (2 * pi * var) ** 0.5
                if step1 / step2 != 0:
                    result *= step1 / step2

            rs_dict[f_key] = result
            
        return rs_dict

    def fit(self, x_train, y_train):
        xdict, ldict, sdict = dict(), dict(), dict()
        for key in set(y_train):
            self.__ft_dict[key] = {}
            sdict[key] = []

        for i in range(len(y_train)):
            ldict[i] = y_train[i]

        for i in ldict.values():
            self.__yi_dict[i] = self.__yi_dict.get(i, 0) + 1

        for i in range(len(x_train[0])):
            xdict[i] = x_train[::, i]

        for i in xdict.values():
            for j in range(len(i)):
                sdict[ldict[j]].append(i[j])

        for k, v in sdict.items():
            m = len(v) // len(x_train[0])
            b, e = 0, m
            for x in range(len(x_train[0])):
                self.__ft_dict[k][x] = {"avg": mean(v[b:e]), "var": pvariance(v[b:e])}
                b += m
                e += m

    def predict(self, x_test):
        y_prim = []
        for x in x_test:
            gauss_prob = self.__gauss_probability(x)
            val_max, key_max = 0, 0
            for k, v in gauss_prob.items():
                if v > val_max:
                    val_max, key_max = v, k

            y_prim.append(key_max)
        return y_prim

    def accuracy(self, y_test, predictions):
        true_predicts = 0
        for i in range(len(y_test)):
            if predictions[i] == y_test[i]:
                true_predicts += 1
        return true_predicts / len(y_test)


x_train, x_test, y_train, y_test = train_test_split(iris().data, iris().target, test_size=0.2, shuffle=True)

nb_model = NaiveBayes()
nb_model.fit(x_train, y_train)
prediction = nb_model.predict(x_test)
print(f"My NaiveBayes Accuracy Score: {nb_model.accuracy(y_test, prediction)*100}%")

sk_model = GaussianNB()
sk_model.fit(x_train, y_train)
prediction = sk_model.predict(x_test)
print(f"Scikit NaiveBayes Accuracy Score: {accuracy_score(y_test, prediction)*100}%")
