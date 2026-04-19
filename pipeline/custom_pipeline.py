import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_custom_pipeline(df):

    df = df.copy()
    df = df.dropna()

    # =========================
    # AUTO FEATURE DETECTION
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # =========================
    # ENCODING
    # =========================
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # KMEANS (AUTO BEST K)
    # =========================
    best_score = -1
    best_k = 2
    best_labels = None
    scores_dict = {}

    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        scores_dict[k] = score

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    df['Cluster'] = best_labels

    return df, best_k, best_score, scores_dict