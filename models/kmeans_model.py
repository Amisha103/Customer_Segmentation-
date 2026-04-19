from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def run_kmeans(X_scaled, n_clusters=5):
  
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    return labels, score



def elbow_method(X_scaled, k_range=range(2, 10)):
 
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)


    plt.figure()
    plt.plot(list(k_range), inertia, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("outputs/elbow_method.png", dpi=300)
    plt.close()

    return inertia