from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def run_dbscan(X, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    unique_labels = set(labels)

    mask = labels != -1

    if len(set(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
    else:
        score = -1

    return labels, score