import matplotlib.pyplot as plt
import os

# Create images folder if not exists
os.makedirs("images", exist_ok=True)


def plot_elbow(wcss):
    plt.figure()
    plt.plot(range(1, len(wcss)+1), wcss, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.savefig("images/elbow.png")
    plt.show()

def plot_feature_comparison(results_df):
    import matplotlib.pyplot as plt

    plt.figure()
    bars = plt.bar(results_df["Features"], results_df["Silhouette Score"])

    # Highlight best bar
    max_index = results_df["Silhouette Score"].idxmax()
    bars[max_index].set_edgecolor('black')
    bars[max_index].set_linewidth(2)

    plt.xticks(rotation=45)
    plt.xlabel("Feature Combinations")
    plt.ylabel("Silhouette Score")
    plt.title("Feature Comparison")

    plt.tight_layout()
    plt.savefig("images/feature_comparison.png")
    plt.show()

def plot_clusters(X_scaled, labels):
    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Cluster Visualization")

    plt.savefig("images/clusters.png")
    plt.show()