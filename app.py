import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from pipeline.default_pipeline import run_default_pipeline
from pipeline.custom_pipeline import run_custom_pipeline

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


    st.subheader("Model Explanation")

    st.write("**Features Used:**")
    st.write(features_used)

    st.write(f"**Best K (KMeans):** {best_k}")
    st.write(f"**Silhouette Score:** {best_score:.3f}")

    st.write(f"**Best DBSCAN Params:** {db_params}")
    st.write(f"**DBSCAN Score:** {db_score:.3f}")


    col1, col2 = st.columns(2)


    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(K_range, inertia, marker='o')
        ax.set_title("Elbow Method")
        st.pyplot(fig)

  
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
        ax2.set_title("Silhouette Scores")
        st.pyplot(fig2)

    st.subheader("KMeans Clusters (Income vs Loyalty)")

    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.scatter(df['income'], df['loyalty_score'], c=labels)
    ax3.set_xlabel("Income")
    ax3.set_ylabel("Loyalty Score")
    st.pyplot(fig3)

 
    st.subheader("DBSCAN Clusters")

    fig4, ax4 = plt.subplots(figsize=(5, 3))
    ax4.scatter(df['income'], df['loyalty_score'], c=db_labels)
    st.pyplot(fig4)

    st.subheader("DBSCAN Details")

    unique_labels = set(db_labels)
    st.write("Clusters found:", len(unique_labels) - (1 if -1 in unique_labels else 0))
    st.write("Noise points:", list(db_labels).count(-1))


    st.subheader("Cluster Summary")
    st.dataframe(df.groupby('Cluster').mean(numeric_only=True))

    st.subheader("Segment Distribution")
    st.bar_chart(df['Cluster'].value_counts())



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
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(list(scores_dict.keys()), list(scores_dict.values()), marker='o')
            ax.set_title("Silhouette Scores")
            st.pyplot(fig)


        st.subheader("Cluster Visualization")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]

            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.scatter(df[x_col], df[y_col], c=df['Cluster'])
            ax3.set_xlabel(x_col)
            ax3.set_ylabel(y_col)
            st.pyplot(fig3)
        else:
            st.warning("Not enough numeric columns to visualize")


        st.subheader("Cluster Summary")
        st.dataframe(df.groupby('Cluster').mean(numeric_only=True))