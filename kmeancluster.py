import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis',edgecolor='k', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids', linewidths=3)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()