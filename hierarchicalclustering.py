from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
x,y =make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=40)
# Perform hierarchical clustering
linkage_matrix = linkage(x, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = cluster.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='rainbow')
plt.title('Agglomerative Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
