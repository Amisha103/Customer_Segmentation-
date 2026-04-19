import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from models.kmeans_model import run_kmeans
from models.dbscan_model import run_dbscan

def run_default_pipeline(df):

    df = df.dropna()
    df = df.drop(columns=['id'])

    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    df_encoded = pd.get_dummies(
        df,
        columns=[
            'education',
            'region',
            'loyalty_status',
            'purchase_frequency',
            'product_category'
        ],
        drop_first=True
    )


    col_loy = [col for col in df_encoded.columns if 'loyalty_status' in col]
    df_encoded['loyalty_score'] = df_encoded[col_loy].sum(axis=1)


    df['loyalty_score'] = df_encoded['loyalty_score']

    features_used = col_loy + ['income', 'loyalty_score']

    X = df_encoded[features_used]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    inertia = []
    K_range = range(2, 8)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)


    best_score = -1
    best_labels = None
    best_k = 2
    scores_dict = {}

    for k in K_range:
        labels, _ = run_kmeans(X_scaled, k)
        score = silhouette_score(X_scaled, labels)

        scores_dict[k] = score

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels


    best_db_score = -1
    best_db_labels = None
    best_params = None

    for eps in [0.2, 0.3, 0.4]:
        for min_samples in [3, 5]:
            labels_db, score_db = run_dbscan(X_scaled, eps, min_samples)

            if score_db > best_db_score:
                best_db_score = score_db
                best_db_labels = labels_db
                best_params = (eps, min_samples)

    df['Cluster'] = best_labels

    return (
        df, features_used, X_scaled,
        best_labels, best_k, best_score,
        scores_dict, inertia, K_range,
        best_db_labels, best_db_score, best_params
    )


