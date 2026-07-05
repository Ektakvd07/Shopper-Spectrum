# Shopper-Spectrum
# Retail Customer Segmentation and Recommendation Streamlit App

## Files

- app.py: Streamlit application
- retail_data.csv: CSV export of the uploaded Excel dataset
- requirements.txt: Python dependencies

## How to run

1. Install dependencies:

   pip install -r requirements.txt

2. Start the app:

   streamlit run app.py

## Features

- Transaction volume by country
- Top-selling products
- Purchase trends over time
- Monetary distribution per transaction and customer
- RFM distributions
- Elbow curve and silhouette score
- KMeans, DBSCAN, and Hierarchical clustering
- Customer cluster profiles and segment labels
- 2D and 3D RFM cluster visualization
- Saved KMeans model for Streamlit usage
- Product recommendation heatmap / similarity matrix
- Top 5 similar product recommendations using item-based collaborative filtering
