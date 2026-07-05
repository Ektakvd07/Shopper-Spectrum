import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.set_page_config(page_title="Retail Customer Segmentation & Recommendation", layout="wide")
st.title("Retail Customer Segmentation & Product Recommendation")
st.caption("EDA, RFM clustering, and item-based product recommendation for retail transaction data.")

DATA_PATH = "online_retail_cleaned.csv"
MODEL_PATH = "best_kmeans_model.pkl"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df = df.dropna(subset=["InvoiceNo", "Description", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["Description"] = df["Description"].astype(str).str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    return df

@st.cache_data
def build_rfm(df):
    latest_date = df["InvoiceDate"].max()
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (latest_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

@st.cache_data
def recommendation_matrix(df):
    matrix = df.pivot_table(index="CustomerID", columns="Description", values="Quantity", aggfunc="sum", fill_value=0)
    similarity = cosine_similarity(matrix.T)
    sim_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
    return matrix, sim_df

def label_segments(clustered):
    profiles = clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    labels = {}
    if profiles.empty:
        return labels
    r33, r66 = profiles["Recency"].quantile([0.33, 0.66])
    f33, f66 = profiles["Frequency"].quantile([0.33, 0.66])
    m33, m66 = profiles["Monetary"].quantile([0.33, 0.66])
    for cluster, row in profiles.iterrows():
        r, f, m = row["Recency"], row["Frequency"], row["Monetary"]
        if r <= r33 and f >= f66 and m >= m66:
            labels[cluster] = "High-Value"
        elif f >= f33 and m >= m33 and r <= r66:
            labels[cluster] = "Regular"
        elif r >= r66 and f <= f33 and m <= m33:
            labels[cluster] = "At-Risk"
        else:
            labels[cluster] = "Occasional"
    return labels

def top_similar_products(product_name, sim_df, top_n=5):
    result = sim_df[product_name].drop(product_name).sort_values(ascending=False).head(top_n)
    return result.reset_index().rename(columns={"Description": "Product", product_name: "Similarity"})

df = load_data(DATA_PATH)
rfm = build_rfm(df)

with st.sidebar:
    st.header("Controls")
    algorithm = st.selectbox("Clustering algorithm", ["KMeans", "DBSCAN", "Hierarchical"])
    max_k = st.slider("Maximum K for elbow/silhouette", 2, 10, 8)
    selected_k = st.slider("Number of clusters for KMeans/Hierarchical", 2, 8, 4)
    eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.0, 0.1)
    min_samples = st.slider("DBSCAN min_samples", 2, 10, 3)

st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{df['InvoiceNo'].nunique():,}")
c2.metric("Customers", f"{df['CustomerID'].nunique():,}")
c3.metric("Products", f"{df['Description'].nunique():,}")
c4.metric("Revenue", f"{df['TotalAmount'].sum():,.2f}")

eda_tab, clustering_tab, recommendation_tab = st.tabs(["EDA", "Clustering Methodology", "Recommendation System"])

with eda_tab:
    st.header("Exploratory Data Analysis")
    country_volume = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False).reset_index(name="Transactions")
    st.plotly_chart(px.bar(country_volume, x="Country", y="Transactions", title="Transaction Volume by Country"), use_container_width=True)

    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(15).reset_index()
    fig_products = px.bar(top_products, x="Quantity", y="Description", orientation="h", title="Top-Selling Products")
    fig_products.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_products, use_container_width=True)

    daily_sales = df.set_index("InvoiceDate").resample("D").agg(Transactions=("InvoiceNo", "nunique"), Revenue=("TotalAmount", "sum")).reset_index()
    st.plotly_chart(px.line(daily_sales, x="InvoiceDate", y=["Transactions", "Revenue"], title="Purchase Trends Over Time"), use_container_width=True)

    txn_amount = df.groupby("InvoiceNo")["TotalAmount"].sum().reset_index(name="TransactionAmount")
    customer_amount = df.groupby("CustomerID")["TotalAmount"].sum().reset_index(name="CustomerMonetary")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(txn_amount, x="TransactionAmount", nbins=30, title="Monetary Distribution per Transaction"), use_container_width=True)
    right.plotly_chart(px.histogram(customer_amount, x="CustomerMonetary", nbins=30, title="Monetary Distribution per Customer"), use_container_width=True)

    st.subheader("RFM Distributions")
    r1, r2, r3 = st.columns(3)
    r1.plotly_chart(px.histogram(rfm, x="Recency", nbins=20, title="Recency"), use_container_width=True)
    r2.plotly_chart(px.histogram(rfm, x="Frequency", nbins=20, title="Frequency"), use_container_width=True)
    r3.plotly_chart(px.histogram(rfm, x="Monetary", nbins=20, title="Monetary"), use_container_width=True)

with clustering_tab:
    st.header("Customer Clustering Methodology")
    st.markdown("""
    **Feature engineering** uses RFM values:
    - **Recency** = latest purchase date in the dataset minus the customer's last purchase date.
    - **Frequency** = number of unique invoices per customer.
    - **Monetary** = total customer spend.

    RFM values are standardized before clustering. KMeans, DBSCAN, and Hierarchical clustering are available.
    """)

    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    k_values = list(range(2, max_k + 1))
    inertias, silhouettes = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled, labels) if len(set(labels)) > 1 else np.nan)

    left, right = st.columns(2)
    left.plotly_chart(px.line(x=k_values, y=inertias, markers=True, labels={"x":"K", "y":"Inertia"}, title="Elbow Curve for Cluster Selection"), use_container_width=True)
    right.plotly_chart(px.line(x=k_values, y=silhouettes, markers=True, labels={"x":"K", "y":"Silhouette Score"}, title="Silhouette Score by K"), use_container_width=True)
    st.info(f"Best K by silhouette score: {k_values[int(np.nanargmax(silhouettes))]}")

    if algorithm == "KMeans":
        model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(scaled)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "features": ["Recency", "Frequency", "Monetary"]}, f)
        st.success(f"KMeans model saved to {MODEL_PATH}")
    elif algorithm == "DBSCAN":
        cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    else:
        cluster_labels = AgglomerativeClustering(n_clusters=selected_k).fit_predict(scaled)

    clustered = rfm.copy()
    clustered["Cluster"] = cluster_labels
    valid = clustered[clustered["Cluster"] != -1]
    labels = label_segments(valid)
    clustered["SegmentLabel"] = clustered["Cluster"].map(labels).fillna("Noise")

    st.subheader("Customer Cluster Profiles")
    profile_table = clustered.groupby(["Cluster", "SegmentLabel"]).agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean")
    ).round(2).reset_index()
    st.dataframe(profile_table, use_container_width=True)

    st.subheader("Cluster Visualizations")
    st.plotly_chart(px.scatter_3d(clustered, x="Recency", y="Frequency", z="Monetary", color="SegmentLabel", hover_data=["CustomerID", "Cluster"], title="3D RFM Customer Segments"), use_container_width=True)
    st.plotly_chart(px.scatter(clustered, x="Recency", y="Monetary", size="Frequency", color="SegmentLabel", hover_data=["CustomerID", "Cluster"], title="RFM Cluster Scatter Plot"), use_container_width=True)

with recommendation_tab:
    st.header("Item-Based Collaborative Filtering")
    st.markdown("The recommender builds a CustomerID-Description matrix and computes cosine similarity between products.")
    matrix, sim_df = recommendation_matrix(df)
    selected_product = st.selectbox("Enter/select product name", sorted(sim_df.index.tolist()))
    st.subheader("Top 5 Similar Products")
    st.dataframe(top_similar_products(selected_product, sim_df), use_container_width=True)

    st.subheader("Product Recommendation Heatmap / Similarity Matrix")
    top_heatmap_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(20).index.tolist()
    heatmap_df = sim_df.loc[top_heatmap_products, top_heatmap_products]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_df, cmap="viridis", ax=ax)
    ax.set_title("Cosine Similarity Matrix for Top Products")
    st.pyplot(fig)
