import matplotlib.pyplot as plt


def plot_kmeans_clusters(X_pca, labels):
    plt.figure(figsize=(5,4))
    plt.scatter(df['income'], df['loyalty_score'], c=labels_kmeans)
    plt.xlabel("Income")
    plt.ylabel("Loyalty Score")
    plt.title("KMeans Clusters (Income vs Loyalty)")
    plt.savefig("outputs/kmeans_scatter_real.png", dpi=300)
    plt.close()


def plot_dbscan_clusters(X_pca, labels):
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("DBSCAN Clusters (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig("outputs/dbscan_clusters.png", dpi=300)
    plt.close()


def plot_silhouette_scores(scores_dict):
    plt.figure()
    ks = list(scores_dict.keys())
    scores = list(scores_dict.values())

    plt.plot(ks, scores, marker='o')
    plt.title("Silhouette Scores vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.savefig("outputs/silhouette_scores.png", dpi=300)
    plt.close()


def plot_segment_distribution(df):
    df['Customer_Segment'].value_counts().plot(kind='bar')
    plt.title("Customer Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/customer_segments_bar.png", dpi=300)
    plt.close()


def plot_segment_pie(df):
    df['Customer_Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Customer Segment Pie Chart")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/customer_segments_pie.png", dpi=300)
    plt.close()