import streamlit as st
import pandas as pd

# Pipelines
from pipeline.default_pipeline import run_default_pipeline
from pipeline.custom_pipeline import run_custom_pipeline

# Visualizations
from visualization.plots import (
    plot_elbow,
    plot_silhouette,
    plot_dbscan,
    plot_dynamic_clusters,
    plot_pca_clusters  
)

st.set_page_config(page_title="Customer Segmentation", layout="wide")


page = st.sidebar.radio(
    "Navigation",
    ["Default Dataset Results", "Upload Your Dataset"]
)

st.title("Customer Segmentation Dashboard")


if page == "Default Dataset Results":

    df = pd.read_csv("dataset/customer_data.csv").sample(n=10000, random_state=42)

    (
        df, features_used, X_scaled,
        labels, best_k, best_score,
        scores_dict, inertia, K_range,
        db_labels, db_score, db_params
    ) = run_default_pipeline(df)


    cluster_summary = df.groupby('Cluster')[['income', 'purchase_amount']].mean()

    sorted_clusters = cluster_summary.sort_values(by='income')

    cluster_names = {}

    cluster_names[sorted_clusters.index[0]] = "Low Value Customers"
    cluster_names[sorted_clusters.index[1]] = "Mid Value Customers"
    cluster_names[sorted_clusters.index[2]] = "High Value Customers"

    df['Segment'] = df['Cluster'].map(cluster_names)


    st.subheader("Model Explanation")

    st.write("**Features Used:**")
    st.write(features_used)

    st.write(f"**Best K (KMeans):** {best_k}")
    st.write(f"**Silhouette Score:** {best_score:.3f}")

    st.write(f"**Best DBSCAN Params:** {db_params}")
    st.write(f"**DBSCAN Score:** {db_score:.3f}")


    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_elbow(K_range, inertia))

    with col2:
        st.pyplot(plot_silhouette(scores_dict))


    st.subheader("Customer Segments (PCA Visualization)")
    st.pyplot(plot_pca_clusters(X_scaled, labels))


    st.subheader("DBSCAN Clusters")
    st.pyplot(plot_dbscan(df, db_labels))

    st.subheader("DBSCAN Details")

    unique_labels = set(db_labels)
    st.write("Clusters found:", len(unique_labels) - (1 if -1 in unique_labels else 0))
    st.write("Noise points:", list(db_labels).count(-1))


    st.subheader("Cluster Summary (With Segment Names)")
    summary = df.groupby(['Cluster', 'Segment']).mean(numeric_only=True)
    st.dataframe(summary)

    st.subheader("Customer Segments Distribution")
    st.bar_chart(df['Segment'].value_counts())

  
    st.subheader("Segment Meaning")
    for k, v in cluster_names.items():
        st.write(f"{k} → {v}")


elif page == "Upload Your Dataset":

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        df, best_k, best_score, scores_dict = run_custom_pipeline(df)


        st.subheader("Model Explanation")

        st.write(f"**Best K:** {best_k}")
        st.write(f"**Silhouette Score:** {best_score:.3f}")


        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_silhouette(scores_dict))


        st.subheader("Cluster Visualization")

        fig = plot_dynamic_clusters(df)

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns to visualize")

     
        st.subheader("Cluster Summary")
        st.dataframe(df.groupby('Cluster').mean(numeric_only=True))