import pandas as pd
import matplotlib.pyplot as plt


# Custom modules
from kmeans_model import run_kmeans
from visualisation import plot_elbow, plot_clusters, plot_feature_comparison

# 1. Load dataset
df = pd.read_csv("dataset/store_customers.csv")

# 2. Data preprocessing
df = df.dropna()
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

# 3. Define feature combinations
feature_sets = {
    "All features": ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "No Gender": ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "No Age": ['Gender', 'Annual Income (k$)', 'Spending Score (1-100)'],
    "Income + Spending": ['Annual Income (k$)', 'Spending Score (1-100)'],
    "Gender + Spending": ['Gender', 'Spending Score (1-100)'],
    "Gender + Age": ['Gender', 'Age'],
    "Age + Income": ['Age', 'Annual Income (k$)']
}

# Store results
results = []

from sklearn.preprocessing import StandardScaler

# 4. Loop through feature sets
for name, features in feature_sets.items():
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run KMeans
    labels, score = run_kmeans(X_scaled, 5)

    results.append((name, score))

# 5. Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Features", "Silhouette Score"])

print("\nFeature Comparison Table:")
print(results_df)

# 6. Save results
results_df.to_csv("results.csv", index=False)

# 7. Plot comparison
plot_feature_comparison(results_df)

# 8. Get BEST feature set
best_features = results_df.loc[results_df["Silhouette Score"].idxmax(), "Features"]

print("\nBest Feature Set:", best_features)

# 9. Train FINAL model using best features
final_features = feature_sets[best_features]
X = df[final_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method (only once for final model)
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plot_elbow(wcss)

# Final KMeans
labels, score = run_kmeans(X_scaled, 5)

df['Cluster'] = labels

print("\nFinal Model Score:", score)

# Plot clusters
plot_clusters(X_scaled, labels)

# Cluster summary
cluster_summary = df.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv("cluster_summary.csv")