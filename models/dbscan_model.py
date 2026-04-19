from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)

    if len(unique_labels) > 1 and -1 not in unique_labels:
        score = silhouette_score(X_scaled, labels)
    else:
        score = -1  

    return labels, score