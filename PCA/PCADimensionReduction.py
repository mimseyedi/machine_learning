from numpy.linalg import svd
from sklearn.datasets import load_iris as iris
from matplotlib.pyplot import scatter, show, plot, title, xlabel, ylabel
from numpy import array, cov
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


class PCADimensionReduction:
    def __init__(self, new_dim):
        self.new_dim = new_dim
        self.percent = None

    def fit_transform(self, data):
        variance_matrix = cov(data.T)
        U, S, _ = svd(variance_matrix)
        new_data = data @ U[:, :self.new_dim]
        percent = sum(S[: self.new_dim]) / sum(S)
        self.percent = percent
        return new_data

    def get_percent(self):
        return self.percent


colors = array(['red', 'green', 'blue'])
data = scale(iris().data)

pca = PCADimensionReduction(new_dim=2)
transformed = pca.fit_transform(data)
scatter(transformed[:, 0], transformed[:, 1], c=colors[iris().target])
show()

# scikit-learn way:
sk_pca = PCA(n_components=2)
transformed = sk_pca.fit_transform(data)
scatter(transformed[:, 0], transformed[:, 1], c=colors[iris().target])
show()
