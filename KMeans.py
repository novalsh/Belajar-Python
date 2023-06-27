from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Membuat dataset acak
X, y = make_blobs(n_samples=200, centers=3, random_state=42)

# Membuat objek K-Means dengan 3 kluster
kmeans = KMeans(n_clusters=3)

# Melakukan clustering
kmeans.fit(X)

# Mendapatkan label kluster
labels = kmeans.labels_

# Menampilkan hasil klustering
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
