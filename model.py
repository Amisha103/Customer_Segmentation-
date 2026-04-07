import pandas as pd

df = pd.read_csv("dataset/customer_data.csv")
df = df.head(10000)
# Basic cleaning
df = df.dropna()

# Drop useless column
df = df.drop(columns=['id'])

# print(df.head())
# print(df.info())

# Label encode Gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# One-hot encode remaining categorical features
df_encoded = pd.get_dummies(
    df,
    columns=['education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category'],
    drop_first=True
)

# print(df_encoded.head())

# print(df_encoded.info())
# print(df_encoded.columns)

from sklearn.preprocessing import StandardScaler

# Separate features
col_freq = [col for col in df_encoded.columns if 'purchase_frequency' in col]
col_edu = [col for col in df_encoded.columns if 'education' in col]
col_reg = [col for col in df_encoded.columns if 'region' in col]
col_prod = [col for col in df_encoded.columns if 'product_category' in col]
col_loy = [col for col in df_encoded.columns if 'loyalty_status' in col]

X = df_encoded[
      col_loy + 
    [  'income'] 
]

# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1, 11):  # trying K from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# # Plot elbow graph
# plt.figure()
# plt.plot(range(1, 11), wcss, marker='o')
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")
# plt.show()

from sklearn.cluster import KMeans

# Train final model
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['Cluster'] = labels

print(df.head())

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)

print("Silhouette Score:", score)