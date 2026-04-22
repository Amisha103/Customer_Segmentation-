import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def save_figure(fig, filename):
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(f"outputs/{filename}", bbox_inches='tight')


def plot_pca_clusters(X_scaled, labels, cluster_names=None):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.7
    )

   
    handles, _ = scatter.legend_elements()

    if cluster_names:
        ax.legend(handles, cluster_names, title="Clusters")
    else:
        unique_labels = sorted(set(labels))
        ax.legend(handles, [f"Cluster {i}" for i in unique_labels])

    ax.set_title("Customer Segmentation (PCA View)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    save_figure(fig, "pca_clusters.png")

    return fig


def plot_elbow(K_range, inertia):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(K_range, inertia, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")

    save_figure(fig, "elbow_plot.png")

    return fig



def plot_silhouette(scores_dict):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
    ax.set_title("Silhouette Scores")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")

    save_figure(fig, "silhouette_plot.png")

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
        name = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
        ax.text(row['income'], row['loyalty_score'], name, fontsize=8)

    ax.set_xlabel("Income")
    ax.set_ylabel("Loyalty Score")
    ax.set_title("Customer Segmentation (KMeans)")
    ax.legend()

    save_figure(fig, "kmeans_clusters.png")

    return fig



def plot_dbscan(X_scaled, db_labels):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=db_labels,
        cmap='viridis',
        alpha=0.7
    )

  
    noise = db_labels == -1
    if sum(noise) > 0:
        ax.scatter(
            X_pca[noise, 0],
            X_pca[noise, 1],
            c='black',
            label='Noise',
            s=20
        )


    unique_labels = sorted(set(db_labels))
    cluster_labels = [
        f"Cluster {i}" if i != -1 else "Noise"
        for i in unique_labels
    ]

    handles, _ = scatter.legend_elements()
    ax.legend(handles, cluster_labels, title="DBSCAN Clusters")

    ax.set_title("DBSCAN Clusters (PCA View)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    save_figure(fig, "dbscan_clusters.png")

    return fig



def plot_dynamic_clusters(df):

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        return None

    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.scatter(
        df[x_col],
        df[y_col],
        c=df['Cluster'],
        alpha=0.6
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Cluster Visualization")

    save_figure(fig, "dynamic_clusters.png")

    return fig