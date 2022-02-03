from random import choices, choice
from numpy import inf, sqrt, array, mean, append
from sklearn.datasets import load_iris as load
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class Kmeans:

    def __init__(self, k=2, max_iter=50, init_method="k-means++"):
        self.__k = k
        self.__max_iter = max_iter
        self.__init_method = init_method
        self.labels_ = []

    def __euclid_distance(self, a, b):
        return sqrt(sum((a - b) ** 2))

    def fit(self, data):
        rows, cols = data.shape
        if self.__init_method == 'random':
            random_mean_points = choices(range(rows), k=self.__k)
            means = data[random_mean_points]
        else:
            means = [data[choice(range(rows))]]
            while len(means) < self.__k:
                distances = list()
                for i in range(rows):
                    min_val = inf
                    for j in range(len(means)):
                        dis = self.__euclid_distance(data[i], means[j])
                        if dis < min_val:
                            min_val = dis
                    distances.append(min_val ** 2)
                t = data[choices(range(rows), weights=distances, k=1)[0]]
                means.append(t)

        costs = list()
        for _ in range(self.__max_iter):
            clusters, distances = array([], dtype=int), list()
            for i in range(rows):
                min_ind, min_val = -1, inf
                for j in range(self.__k):
                    dis = self.__euclid_distance(data[i], means[j])
                    if dis < min_val:
                        min_ind, min_val = j, dis
                clusters = append(clusters, [min_ind], axis=0)
                distances.append(min_val)

            for i in range(self.__k):
                cluster_members = clusters == i
                means[i] = mean(data[cluster_members], axis=0)
            costs.append(sum(distances))

        self.labels_ = clusters


data = load().data

model = Kmeans(k=3, max_iter=100)
model.fit(data)
print("My Kmeans Silhouette Score:", silhouette_score(data, model.labels_))

# scikit-learn way:
s_model = KMeans(n_clusters=3, max_iter=100)
s_model.fit(data)
print("Scikit Kmeans Silhouette Score:", silhouette_score(data, s_model.labels_))
