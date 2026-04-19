import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca_clusters(X_scaled, labels):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7,5))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.7
    )

    ax.set_title("Customer Segmentation (PCA View)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    return fig


def plot_elbow(K_range, inertia):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(K_range, inertia, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    return fig


def plot_silhouette(scores_dict):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
    ax.set_title("Silhouette Scores")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    return fig



def plot_customer_segments(df, labels, cluster_names):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        df['income'],
        df['loyalty_score'],
        c=labels,
        cmap='viridis',
        alpha=0.6
    )

  
    centroids = df.groupby('Cluster')[['income', 'loyalty_score']].mean()

    ax.scatter(
        centroids['income'],
        centroids['loyalty_score'],
        c='red',
        marker='X',
        s=200,
        label='Centroids'
    )

  
    for i, row in centroids.iterrows():
        ax.text(row['income'], row['loyalty_score'], cluster_names[i], fontsize=8)

    ax.set_xlabel("Income")
    ax.set_ylabel("Loyalty Score")
    ax.set_title("Customer Segmentation (KMeans)")
    ax.legend()

    return fig



def plot_dbscan(df, db_labels):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df['income'], df['loyalty_score'], c=db_labels, alpha=0.6)
    ax.set_title("DBSCAN Clusters")
    return fig


def plot_dynamic_clusters(df):

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        return None

    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df[x_col], df[y_col], c=df['Cluster'], alpha=0.6)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Cluster Visualization")

    return fig