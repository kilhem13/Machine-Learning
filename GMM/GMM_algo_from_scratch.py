import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, N, J, X_train):
        self.nbr_clusters = J
        self.nbr_samples = N
        self.Prob_x = np.zeros(X_train.shape)
        self.weight = np.ones(J)/J
        self.means = np.zeros([J, N])
        self.covariances = [np.cov(X_train.T) for _ in range(J)]#np.zeros([N, N, J])
        self.X_train = X_train
    def M_step(self):
        sum_p = []
        for j in range(self.nbr_clusters):
            sum_p.append(sum(self.Prob_x[:][j]))
            self.means[j] = (self.Prob_x[:][j] * self.weight[j]).sum(axis=0)/sum(sum_p)
            for f in self.nbr_samples:
                

        self.weight =  sum_p/sum(sum_p)


    def E_step(self, X):
        likelihood = []
        x_norm = np.zeros([len(X), self.nbr_clusters])
        for j in range(len(X)):
            for i in range(self.nbr_clusters):
                x_norm[j][i] = (self.Gaussian_fonction(X[j], self.means[i], self.covariances[i]))
        self.Prob_x = (x_norm * self.weight)/(x_norm * self.weight).sum(axis=0)

    def Gaussian_fonction(self, x, mu, sig):
        return (np.exp(-0.5 * np.matmul(np.matmul((x - mu).T, np.linalg.inv(sig)),(x - mu)))) /np.sqrt(np.power(2 * np.pi, self.nbr_samples)*np.linalg.det(sig))





n_samples = 300
# generate random sample, two components
np.random.seed(0)
# generate spherical data centered on (20, 20)

shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
shifted_gaussian_2 = np.random.randn(n_samples, 2) + np.array([7, 9])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian, shifted_gaussian_2])
# fit a Gaussian Mixture Model with two components
plt.scatter(X_train[:, 0], X_train[:, 1], .8)
plt.axis('tight')
plt.show()
print(len(X_train))
gmm = GMM(2, 3, X_train)
gmm.E_step(X_train)
gmm.M_step()
