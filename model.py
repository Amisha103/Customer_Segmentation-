import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Custom modules
from kmeans_model import run_kmeans
from dbscan_model import run_dbscan
from visualisation import plot_elbow, plot_clusters, plot_feature_comparison

# 1. Load dataset
df = pd.read_csv("dataset/store_customers.csv")

# 2. Data preprocessing
df = df.dropna()
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

# 3. Feature combinations
feature_sets = {
    "All features": ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "No Gender": ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "No Age": ['Gender', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "Income + Spending": ['Annual Income (k$)', 'Spending Score (1-100)'],
    "Gender + Spending": ['Gender', 'Spending Score (1-100)'],
    "Gender + Age": ['Gender', 'Age'],
    "Age + Income": ['Age', 'Annual Income (k$)']
}

# =========================
# 🔵 KMEANS EXPERIMENT
# =========================

kmeans_results = []

for name, features in feature_sets.items():
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels, score = run_kmeans(X_scaled, 5)
    kmeans_results.append((name, score))

kmeans_df = pd.DataFrame(kmeans_results, columns=["Features", "Silhouette Score"])

print("\nKMeans Results:")
print(kmeans_df)

kmeans_df.to_csv("kmeans_results.csv", index=False)
plot_feature_comparison(kmeans_df)

# Best KMeans model
best_kmeans_row = kmeans_df.loc[kmeans_df["Silhouette Score"].idxmax()]
best_kmeans_features = best_kmeans_row["Features"]
best_kmeans_score = best_kmeans_row["Silhouette Score"]

print("\nBest KMeans Model:")
print(best_kmeans_row)

# =========================
# 🟢 DBSCAN EXPERIMENT
# =========================

dbscan_results = []

for name, features in feature_sets.items():
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels, score = run_dbscan(X_scaled)
    dbscan_results.append((name, score))

dbscan_df = pd.DataFrame(dbscan_results, columns=["Features", "Silhouette Score"])

print("\nDBSCAN Results:")
print(dbscan_df)

dbscan_df.to_csv("dbscan_results.csv", index=False)
plot_feature_comparison(dbscan_df)

# Best DBSCAN model
best_dbscan_row = dbscan_df.loc[dbscan_df["Silhouette Score"].idxmax()]
best_dbscan_features = best_dbscan_row["Features"]
best_dbscan_score = best_dbscan_row["Silhouette Score"]

print("\nBest DBSCAN Model:")
print(best_dbscan_row)

# =========================
# 🔥 FINAL COMPARISON
# =========================

print("\nFINAL COMPARISON:")
print("Best KMeans Score:", best_kmeans_score)
print("Best DBSCAN Score:", best_dbscan_score)

if best_kmeans_score > best_dbscan_score:
    final_algorithm = "KMeans"
    final_features = best_kmeans_features
    final_score = best_kmeans_score
else:
    final_algorithm = "DBSCAN"
    final_features = best_dbscan_features
    final_score = best_dbscan_score

print("\nFINAL SELECTED MODEL:")
print("Algorithm:", final_algorithm)
print("Features:", final_features)
print("Score:", final_score)



X = df[feature_sets[final_features]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


if final_algorithm == "KMeans":
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plot_elbow(wcss)


if final_algorithm == "KMeans":
    labels, score = run_kmeans(X_scaled, 5)
else:
    labels, score = run_dbscan(X_scaled)

df['Cluster'] = labels

print("\nFinal Model Score:", score)


plot_clusters(X_scaled, labels)


cluster_summary = df.groupby('Cluster').mean()

print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv("cluster_summary.csv")