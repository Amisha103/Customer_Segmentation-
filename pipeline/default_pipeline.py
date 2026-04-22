import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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


    dbscan_features = col_loy + [ 'income', 'age']
    X_db = df_encoded[dbscan_features] 

   

    scaler_db = StandardScaler()
    X_db_scaled = scaler_db.fit_transform(X_db)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_db_scaled)


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

    for eps in np.arange(0.2, 1.0, 0.1):
        for min_samples in [3, 5, 7]:

            labels_db, score_db, noise_ratio, noise_points = run_dbscan(
                X_pca, eps, min_samples  
            )

            labels_db = np.array(labels_db)

            n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)



            if (
                n_clusters >= 2 and
                n_clusters <= 6 and
                noise_ratio < 0.4 and
                score_db > best_db_score
            ):
                best_db_score = score_db
                best_db_labels = labels_db
                best_params = (eps, min_samples)


    df['Cluster'] = best_labels 


    return (
        df,
        features_used,
        X_scaled,
        best_labels,
        best_k,
        best_score,
        scores_dict,
        inertia,
        K_range,
        best_db_labels,
        best_db_score,
        best_params
    )