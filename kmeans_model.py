from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(X_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    return labels, score