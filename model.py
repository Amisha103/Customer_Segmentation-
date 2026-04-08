import pandas as pd
from sklearn.preprocessing import StandardScaler

# Custom modules
from kmeans_model import run_kmeans
from dbscan_model import run_dbscan

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset/customer_data.csv")

# Use sample (for performance)
df = df.sample(n=10000, random_state=42)

# =========================
# 2. PREPROCESSING
# =========================
df = df.dropna()
df = df.drop(columns=['id'])

# Encode gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# One-hot encoding
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

X = df_encoded[
    col_loy +
    ['income']
]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


labels_kmeans, score_kmeans = run_kmeans(X_scaled, 5)

print("\nKMeans Silhouette Score:", score_kmeans)


best_dbscan_score = -1
best_params = None
best_labels_dbscan = None

for eps in [0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 10]:
        labels_db, score_db = run_dbscan(X_scaled, eps, min_samples)

        print(f"DBSCAN eps={eps}, min_samples={min_samples} → score={score_db}")

        if score_db > best_dbscan_score:
            best_dbscan_score = score_db
            best_params = (eps, min_samples)
            best_labels_dbscan = labels_db

print("\nBest DBSCAN Params:", best_params)
print("Best DBSCAN Score:", best_dbscan_score)


print("\nFINAL COMPARISON:")
print("KMeans Score:", score_kmeans)
print("DBSCAN Score:", best_dbscan_score)

if score_kmeans > best_dbscan_score:
    final_model = "KMeans"
    final_labels = labels_kmeans
    final_score = score_kmeans
else:
    final_model = "DBSCAN"
    final_labels = best_labels_dbscan
    final_score = best_dbscan_score

print("\nFINAL SELECTED MODEL:", final_model)
print("Final Score:", final_score)

# =========================
# 5. SAVE RESULTS
# =========================
df['Cluster'] = final_labels

# Cluster summary
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)

print("\nCluster Summary:")
print(cluster_summary)

# Save outputs
cluster_summary.to_csv("outputs/cluster_summary.csv")
df.to_csv("outputs/final_clustered_data.csv", index=False)

cluster_names = {}

avg_income = cluster_summary['income'].mean()
avg_satisfaction = cluster_summary['satisfaction_score'].mean()
avg_promo = cluster_summary['promotion_usage'].mean()
for cluster in cluster_summary.index:
    row = cluster_summary.loc[cluster]

    income = row['income']
    satisfaction = row['satisfaction_score']
    promo = row['promotion_usage']

    if income > avg_income and satisfaction > avg_satisfaction:
        cluster_names[cluster] = "Premium Customers 💎"
    
    elif income < avg_income and satisfaction < avg_satisfaction:
        cluster_names[cluster] = "Budget Customers 💸"
    
    elif promo > avg_promo:
        cluster_names[cluster] = "Deal Hunters 🏷️"
    
    elif satisfaction > avg_satisfaction:
        cluster_names[cluster] = "Happy Customers 😊"
    
    else:
        cluster_names[cluster] = "Regular Customers 👥"

df['Customer_Segment'] = df['Cluster'].map(cluster_names)
print("\nCluster Meaning:")

for k, v in cluster_names.items():
    print(f"Cluster {k} → {v}")