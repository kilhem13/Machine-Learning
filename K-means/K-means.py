import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import math

index = 0

class K_Means:

    def __init__(self, k =4, tolerance = 0.0001, max_iterations = 10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = []
        self.diff_class = []
        self.index = 0

    def Eucl_dist(self, p, q):
        distance = 0
        #print( "\n" + str(q) + "\n" + str(self.index))
        for i in range(len(p)):
            distance += (p[i]-q[i])**2
        self.index += 1/3
        return math.sqrt(distance)

    def find_clusters(self, X):
        for i in range(self.k):
            self.centroids.append(X[i])
            self.diff_class.append([])
        for j in range(100):
            for point in X:
                centroids_dist = [self.Eucl_dist(point, centroid) for centroid in self.centroids]
                self.diff_class[centroids_dist.index(min(centroids_dist))].append(point)

            #calculate the average point of each cluster
            for diff_class in self.diff_class:
                self.centroids[self.diff_class.index(diff_class)] = np.average(diff_class, axis=0)


        return self.centroids


# Generate some data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

k = K_Means()
k.find_clusters(X)
for centroid in k.centroids:
    plt.scatter(centroid[0], centroid[1], s = 100, marker="X");
plt.show()
