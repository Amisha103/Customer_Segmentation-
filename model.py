import pandas as pd
from sklearn.preprocessing import StandardScaler

from kmeans_model import run_kmeans

# Load data
df = pd.read_csv("dataset/customer_data.csv")

# Sampling
df = df.sample(n=10000, random_state=42)

# Preprocessing
df = df.dropna()
df = df.drop(columns=['id'])

# Encoding
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

df_encoded = pd.get_dummies(
    df,
    columns=['education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category'],
    drop_first=True
)

# Feature selection (FINAL)
col_loy = [col for col in df_encoded.columns if 'loyalty_status' in col]

X = df_encoded[
    col_loy +
    ['income']
]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run KMeans
labels, score = run_kmeans(X_scaled, 5)

df['Cluster'] = labels

print("KMeans Silhouette Score:", score)