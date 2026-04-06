import pandas as pd
from visualisation import plot_elbow, plot_feature_comparison, plot_clusters
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

# Elbow method
wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plot_elbow(wcss)   


# Final model
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

plot_clusters(X_scaled, labels)  


# Evaluation
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)



results = [
    ("All features", 0.459),
    ("No Gender", 0.353),
    ("No Age", 0.474),
    ("Income + Spending", 0.354),
    ("Gender + Spending", 0.578),
    ("Gender + Age", 0.603),
    ("Age + Income", 0.439)
]

results_df = pd.DataFrame(results, columns=["Features", "Silhouette Score"])

print("\nFeature Comparison Table:")
print(results_df)

best_row = results_df.loc[results_df["Silhouette Score"].idxmax()]

print("\nBest Feature Combination:")
print(best_row)

results_df.to_csv("results.csv", index=False)

plot_feature_comparison(results_df)