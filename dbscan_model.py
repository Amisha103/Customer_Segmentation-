from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Check if valid clustering exists
    unique_labels = set(labels)

    # If only one cluster or all noise → invalid
    if len(unique_labels) <= 1 or (len(unique_labels) == 1 and -1 in unique_labels):
        score = -1
    else:
        score = silhouette_score(X_scaled, labels)

    return labels, score