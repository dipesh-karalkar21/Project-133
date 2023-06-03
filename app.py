from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

df = pd.read_csv("final.csv")

X = df.iloc[:, [5, 6]].values

print(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

plt.figure(figsize=(15,7))
sns.scatterplot(x = X[y_kmeans == 0, 0], y = X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(x = X[y_kmeans == 1, 0], y = X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(x = X[y_kmeans == 2, 0], y = X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(x = X[y_kmeans == 3, 0], y = X[y_kmeans == 3, 1], color = 'red', label = 'Cluster 4')
sns.scatterplot(x = kmeans.cluster_centers_[:, 0], y = kmeans.cluster_centers_[:, 1], color = 'black', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.show()