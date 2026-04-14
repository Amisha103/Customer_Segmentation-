import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from kmeans_model import run_kmeans
from dbscan_model import run_dbscan

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["📊 Default Dataset Results", "📂 Upload Your Dataset"]
)

st.title("📊 Customer Segmentation Dashboard")

# =========================
# PIPELINE FUNCTION
# =========================
def run_pipeline(df):

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

    # =========================
    # FEATURE SELECTION
    # =========================
    col_loy = [col for col in df_encoded.columns if 'loyalty_status' in col]
    df_encoded['loyalty_score'] = df_encoded[col_loy].sum(axis=1)

# ✅ IMPORTANT FIX
    df['loyalty_score'] = df_encoded['loyalty_score']

    features_used = col_loy + ['income', 'loyalty_score']

    X = df_encoded[features_used]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # ELBOW METHOD
    # =========================
    inertia = []
    K_range = range(2, 8)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    # =========================
    # KMEANS
    # =========================
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

    # =========================
    # DBSCAN
    # =========================
    best_db_score = -1
    best_db_labels = None
    best_params = None

    for eps in [0.2, 0.3, 0.4]:
        for min_samples in [3, 5]:
            labels_db, score_db = run_dbscan(X_scaled, eps, min_samples)

            if score_db > best_db_score:
                best_db_score = score_db
                best_db_labels = labels_db
                best_params = (eps, min_samples)

    df['Cluster'] = best_labels

    return (
        df, features_used, X_scaled,
        best_labels, best_k, best_score,
        scores_dict, inertia, K_range,
        best_db_labels, best_db_score, best_params
    )


# =========================
# 📊 PAGE 1
# =========================
if page == "📊 Default Dataset Results":

    df = pd.read_csv("dataset/customer_data.csv").sample(n=10000, random_state=42)

    (
        df, features_used, X_scaled,
        labels, best_k, best_score,
        scores_dict, inertia, K_range,
        db_labels, db_score, db_params
    ) = run_pipeline(df)

    # =========================
    # INFO SECTION
    # =========================
    st.subheader("📌 Model Explanation")

    st.write("**Features Used:**")
    st.write(features_used)

    st.write(f"**Best K (KMeans):** {best_k}")
    st.write(f"**Silhouette Score:** {best_score:.3f}")

    st.write(f"**Best DBSCAN Params:** {db_params}")
    st.write(f"**DBSCAN Score:** {db_score:.3f}")

    # =========================
    # VISUALS
    # =========================
    col1, col2 = st.columns(2)

    # 🔹 Elbow
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(K_range, inertia, marker='o')
        ax.set_title("Elbow Method")
        st.pyplot(fig)

    # 🔹 Silhouette
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
        ax2.set_title("Silhouette Scores")
        st.pyplot(fig2)

    # 🔹 Scatter (REAL FEATURES)
    st.subheader("🎯 KMeans Clusters (Income vs Loyalty)")

    fig3, ax3 = plt.subplots(figsize=(5,3))
    ax3.scatter(df['income'], df['loyalty_score'], c=labels)
    ax3.set_xlabel("Income")
    ax3.set_ylabel("Loyalty Score")
    st.pyplot(fig3)

    # 🔹 DBSCAN Plot
    st.subheader("🔍 DBSCAN Clusters")

    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.scatter(df['income'], df['loyalty_score'], c=db_labels)
    st.pyplot(fig4)

    # =========================
    # DBSCAN DETAILS
    # =========================
    st.subheader("DBSCAN Details")

    unique_labels = set(db_labels)
    st.write("Clusters found:", len(unique_labels) - (1 if -1 in unique_labels else 0))
    st.write("Noise points:", list(db_labels).count(-1))

    # =========================
    # SUMMARY
    # =========================
    st.subheader("📊 Cluster Summary")
    st.dataframe(df.groupby('Cluster').mean(numeric_only=True))

    st.subheader("📊 Segment Distribution")
    st.bar_chart(df['Cluster'].value_counts())


# =========================
# 📂 PAGE 2
# =========================
elif page == "📂 Upload Your Dataset":

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        (
            df, features_used, X_scaled,
            labels, best_k, best_score,
            scores_dict, inertia, K_range,
            db_labels, db_score, db_params
        ) = run_pipeline(df)

        st.subheader("📌 Model Explanation")

        st.write("**Features Used:**")
        st.write(features_used)

        st.write(f"**Best K:** {best_k}")
        st.write(f"**Silhouette Score:** {best_score:.3f}")

        # VISUALS
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(K_range, inertia, marker='o')
            ax.set_title("Elbow Method")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
            ax2.set_title("Silhouette Scores")
            st.pyplot(fig2)

        st.subheader("🎯 Clusters")

        fig3, ax3 = plt.subplots(figsize=(5,3))
        ax3.scatter(df['income'], df['loyalty_score'], c=labels)
        st.pyplot(fig3)

        st.subheader("📊 Cluster Summary")
        st.dataframe(df.groupby('Cluster').mean(numeric_only=True))