import pandas as pd

df = pd.read_csv("dataset/store_customers.csv")
df = df.dropna()
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# print(df.isnull().sum())
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
