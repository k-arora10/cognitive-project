import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

df = pd.read_csv("/Users/manavjotsingh/pythonProject/Mall_Customers.csv.xls")

# Selecting relevant columns for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df = df[features].dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Autoencoder Model
input_dim = df_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation='relu')(input_layer)
encoded = Dense(3, activation='relu')(encoded)
decoded = Dense(6, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(df_scaled, df_scaled, epochs=100, batch_size=8, verbose=0)

# Extract the Encoder Model
encoder = Model(input_layer, encoded)
df_encoded = encoder.predict(df_scaled)

# Streamlit UI Code
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title('Customer Segmentation App')

with st.sidebar:
    selected = option_menu("Navigation", ["Home", "Upload & Predict"], icons=["house", "upload"], menu_icon="cast",
                           default_index=0)
    num_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10, value=3)

# KMeans Clustering on Encoded Data
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
labels = kmeans.fit_predict(df_encoded)
df['Cluster'] = labels

# Compute Clustering Scores
silhouette_avg = silhouette_score(df_encoded, labels)
inertia = kmeans.inertia_

# Saving the KMeans model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

if selected == "Home":
    st.header('Overview')
    st.write("""
    Welcome to the Customer Segmentation App!  
    This application uses Autoencoders and KMeans Clustering to segment customers based on:
    - Age  
    - Annual Income  
    - Spending Score  

    Features:
    - Automatically groups customers into different clusters  
    - Visualize customer segments with scatter plots  
    - Upload your own dataset and get predictions  
    - Clustering Metrics like Silhouette Score & Inertia  
    - Cluster Summary Table to understand customer groups  
    """)

    # Displaying Clustering Scores
    st.write(f'Silhouette Score (Higher is Better): {silhouette_avg:.4f}')
    st.write(f'Inertia (WCSS - Lower is Better): {inertia:.2f}')

    # Example Scatter Plot
    st.write('Example: Customer Segments Visualization')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis',
                    ax=ax)
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    st.pyplot(fig)
    plt.clf()

    # Example Cluster Summary Table
    st.write("Example: Cluster Summary Table")
    example_summary = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Customers": [15, 20, 10],
        "Avg_Age": [32.5, 45.3, 28.7],
        "Avg_Income ($K)": [50.2, 75.8, 22.3],
        "Avg_Spending_Score": [65.1, 30.2, 80.5]
    })
    st.dataframe(example_summary)

if selected == "Upload & Predict":
    st.header('Upload Your Data & Predict')
    uploaded_file = st.file_uploader('Upload File', type=['csv'])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)

        if set(features).issubset(user_data.columns):
            # Scaling and encoding the uploaded data
            user_data_scaled = scaler.transform(user_data[features])
            user_data_encoded = encoder.predict(user_data_scaled)

            # Predict clusters using the trained model
            user_data['Cluster'] = kmeans.predict(user_data_encoded)

            # Compute silhouette score for the uploaded dataset
            if len(set(user_data['Cluster'])) > 1:  # Avoid silhouette score error with 1 cluster
                user_silhouette_score = silhouette_score(user_data_encoded, user_data['Cluster'])
            else:
                user_silhouette_score = "Not available (only 1 cluster detected)"

            # Display user input dataset
            st.write('Uploaded Dataset:')
            st.dataframe(user_data)

            # Display cluster summary with averages
            cluster_summary = user_data.groupby('Cluster').agg(
                Customers=('Cluster', 'count'),
                Avg_Age=('Age', 'mean'),
                Avg_Income=('Annual Income (k$)', 'mean'),
                Avg_Spending_Score=('Spending Score (1-100)', 'mean')
            ).reset_index()

            st.write('Cluster Summary (Averages):')
            st.dataframe(cluster_summary)

            # Display silhouette score
            st.write(f'Silhouette Score for Uploaded Data: {user_silhouette_score}')

            # Visual Representation of the clustered data
            st.write('Visual Representation of Segmented Data:')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=user_data['Annual Income (k$)'], y=user_data['Spending Score (1-100)'],
                            hue=user_data['Cluster'], palette='viridis', ax=ax)
            plt.title('Uploaded Data Customer Segments')
            plt.xlabel('Annual Income (k$)')
            plt.ylabel('Spending Score (1-100)')
            st.pyplot(fig)
            plt.clf()

        else:
            st.error(f'Missing columns! Ensure your file contains: {features}')
