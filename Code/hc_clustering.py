from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

# using dendrogram to find optimal no of clusters
# ward - method for min the variance inside clusters
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


# train hc model on dataset
hc = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)

print(y_hc)


# visualize clustering
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1],
            s=100, c='magenta', label='Cluster 5')


plt.title("Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
