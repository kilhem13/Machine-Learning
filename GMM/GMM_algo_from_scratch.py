import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data


class GMM:
    def __init__(self,K, X_train):
        self.data_shape = X_train.shape
        self.nbr_clusters = K
        self.nbr_samples, self.nbr_features = self.data_shape
        self.Prob_x = np.zeros(X_train.shape)
        self.weight = np.ones(K)/K
        self.var = np.ones(3)
        self.means = [] # np.zeros([K, self.nbr_features])
        self.covariances = [np.cov(X_train.T) for _ in range(K)]#np.zeros([N, N, J])
        for i in range(K):
            self.means.append(X_train[i])
        self.X_train = X_train
    def M_step(self):
        sum_p = []
        mu = []
        for i in range(3):
            mu.append(0)
        for j in range(self.nbr_clusters):
            weight = self.Prob_x[:, [j]]
            total_weight = weight.sum()
            mu[j] = (self.X_train * weight).sum(axis=0) / total_weight

            sum_p.append(sum(self.Prob_x[:][j]))
            self.means[j] = sum(self.X_train * weight) / sum(self.Prob_x[i][j] for i in range(self.nbr_samples))
            self.covariances[j] = np.diag(sum(self.Prob_x[i][j] * (self.X_train[i] - mu[j])**2 for i in range(self.nbr_samples)) / sum(self.Prob_x[i][j] for i in range(self.nbr_samples)))

            #self.covariances[j] = np.dot((sum_p[j] * (X_train - self.means[j])).T, (X_train - self.means[j])) / sum(self.Prob_x[i][j] for i in range(self.nbr_samples))
            #for i in range(len(X_train)):
                #self.covariances[j] += ((1/sum(sum_p)))
            #for f in self.nbr_samples:


        self.weight =  self.Prob_x.mean(axis=0)


    def E_step(self, X):
        likelihood = np.zeros((self.nbr_samples, self.nbr_clusters))
        x_norm = np.zeros([len(X), self.nbr_clusters])
        for j in range(len(X)):
            for i in range(self.nbr_clusters):
                x_norm[j][i] = (self.Gaussian_fonction(X[j], self.means[i], self.covariances[i]))
        self.Prob_x = (x_norm * self.weight)/(x_norm * self.weight).sum(axis=1)[:, np.newaxis]

    def Gaussian_fonction(self, x, mu, sig):
        #return (np.exp(-0.5 * np.matmul((x - mu), (x - mu))/sig))/np.sqrt((2*np.pi)*sig)
        res = (np.exp(-(0.5 / (sig)) *
                      np.matmul((x - mu).T, (x - mu)))) \
              / np.sqrt(np.power(2 * np.pi, self.nbr_features) * np.linalg.det(sig))
        res2 = (np.exp(-0.5 * np.matmul(np.matmul((x - mu).T,np.linalg.inv(sig)), (x - mu)))) / np.sqrt(np.power(2 * np.pi, self.nbr_features)*np.linalg.det(sig))
        return res[0][0]



n_samples = 300
np.random.seed(0)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
shifted_gaussian_2 = np.random.randn(n_samples, 2) + np.array([7, 9])
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
X_train = np.vstack([shifted_gaussian, stretched_gaussian, shifted_gaussian_2])
#plt.scatter(X_train[0:299, 0], X_train[0:299, 1], .8, edgecolors='blue')
#plt.scatter(X_train[600:899, 0], X_train[600:899, 1], .8, edgecolors='green')
#plt.scatter(X_train[300:599, 0], X_train[300:599, 1], .8, edgecolors='red')
#plt.axis('tight')
#plt.show()
print(len(X))
gmm = GMM(3, X)
for i in range(20):
    gmm.E_step(X)
    gmm.M_step()
