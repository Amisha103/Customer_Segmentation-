import pandas as pd
# 1. Load the dataset
df = pd.read_csv("dataset/store_customers.csv")
# 2. Data preprocessing
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# print(df.isnull().sum())
df = df.dropna()
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

X = df[['Gender', 'Age']]

#Data Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1, 10), wcss)
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")
# plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)


df['Cluster'] = labels

print(df.head())

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)