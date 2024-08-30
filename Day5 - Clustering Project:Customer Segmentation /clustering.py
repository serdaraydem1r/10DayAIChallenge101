#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
#%%
df = pd.read_csv('/Users/fsa/PycharmProjects/10 Day AI Challenge 101/Day5 - Clustering Project:Customer Segmentation /Mall_Customers.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isna().sum())

#%%
plt.figure(figsize=(10,10))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(10,10))
sns.scatterplot(x='Age',y='Spending Score (1-100)', hue='Gender', data=df)
plt.title("Age vs Spending Score")
plt.show()

plt.figure(figsize=(10,10))
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)', hue='Gender', data=df)
plt.title("Anual Income vs Spending Score")
plt.show()

#%% K-means
X = df[['Annual Income (k$)','Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range (1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(10,10))
plt.plot(range(1,11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(10,10))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis')
plt.title("K-means Clustering Report")
plt.show()

#%%
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Örnek Sayısı')
plt.ylabel('Uzaklık')
plt.show()

agg_clustering = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
df['Hierarchical Cluster'] = agg_clustering.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, hue='Hierarchical Cluster', palette='viridis')
plt.title("Hiyerarşik Kümeleme Sonuçları")
plt.show()

#%%
kmeans_silhouette = silhouette_score(X_scaled, df['Cluster'])
print(f"K-means Silhouette Skoru: {kmeans_silhouette}")

hierarchical_silhouette = silhouette_score(X_scaled, df['Hierarchical Cluster'])