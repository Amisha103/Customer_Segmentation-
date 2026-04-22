from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np


def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering and returns:
    - labels
    - silhouette score (excluding noise)
    - noise ratio
    - number of noise points
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    labels = np.array(labels)

    # =========================
    # NOISE CALCULATION
    # =========================
    noise_points = np.sum(labels == -1)
    noise_ratio = noise_points / len(labels)

    # =========================
    # REMOVE NOISE FOR SCORING
    # =========================
    mask = labels != -1

    # If not enough clusters → invalid
    if len(set(labels[mask])) < 2:
        return labels, -1, noise_ratio, noise_points

    # =========================
    # SILHOUETTE SCORE
    # =========================
    try:
        score = silhouette_score(X[mask], labels[mask])
    except:
        score = -1

    return labels, score, noise_ratio, noise_points