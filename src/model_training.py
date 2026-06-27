import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

sns.set_style("whitegrid")

def preprocess_and_scale(customers, feature_cols, skewed_cols):
    """Log transforms and scales features."""
    X = customers[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    for col in skewed_cols:
        X[col] = np.log1p(X[col])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_scaled, scaler

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Evaluates K-Means for different K values."""
    print("Evaluating optimal K...")
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    
    os.makedirs("outputs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, "bo-"); ax1.set_title("Elbow Method")
    ax2.plot(k_range, sil_scores, "ro-"); ax2.axvline(x=best_k, color="green", linestyle="--"); ax2.set_title("Silhouette Score")
    plt.savefig("outputs/04_optimal_k.png", dpi=150, bbox_inches="tight"); plt.close()
    return best_k

def train_final_model(X_scaled, customers, feature_cols, k=4):
    """Trains K-Means, then enhances with GMM Soft Clustering."""
    print(f"Training final K-Means model (K={k})...")
    
    # 1. K-MEANS (Hard Clustering)
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    customers["cluster"] = km.fit_predict(X_scaled)
    joblib.dump(km, "models/kmeans_model.pkl")
    
    # Lock Segment Names (Based on latest run)
    segment_names = {
        0: "Churning At-Risk", 
        1: "Loyal High-Spenders", 
        2: "New Customers", 
        3: "Steady Mid-Value"
    }
    customers["segment"] = customers["cluster"].map(segment_names)

    # 2. GMM (Soft Clustering - Probabilities)
    print("Training GMM for soft probabilities...")
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    joblib.dump(gmm, "models/gmm_model.pkl")
    
    # Get probabilities
    probs = gmm.predict_proba(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    
    # --- SMART 1-to-1 MAPPING ---
    # Create an overlap matrix: How many customers are in GMM_i and KM_j?
    overlap = np.zeros((k, k), dtype=int)
    for gmm_c in range(k):
        for km_c in range(k):
            overlap[gmm_c, km_c] = np.sum((gmm_labels == gmm_c) & (customers["cluster"] == km_c))
            
    # Greedy 1-to-1 mapping (prevents overwriting)
    gmm_to_km_map = {}
    used_km = set()
    
    # Sort GMM components by their strongest overlap first
    for gmm_c in np.argsort(-overlap.max(axis=1)):
        best_km = np.argmax(overlap[gmm_c])
        # If that K-Means cluster is already taken, find the next best one
        while best_km in used_km:
            overlap[gmm_c, best_km] = -1 # Temporarily set to -1 so argmax ignores it
            best_km = np.argmax(overlap[gmm_c])
            
        gmm_to_km_map[gmm_c] = best_km
        used_km.add(best_km)

    # Assign probabilities using the safe 1-to-1 map
    for gmm_idx, km_idx in gmm_to_km_map.items():
        seg_name = segment_names[km_idx]
        customers[f"prob_{seg_name}"] = probs[:, gmm_idx].round(3)

    # 3. Profiling
    profile = customers.groupby("segment").agg(
        count=("CustomerID", "size"), avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean")
    ).round(2)
    profile["revenue_share_%"] = (profile["avg_monetary"] * profile["count"] / (profile["avg_monetary"] * profile["count"]).sum() * 100).round(1)
    
    print("\nSegment Summary:")
    print(profile[["count", "avg_recency", "avg_frequency", "avg_monetary", "revenue_share_%"]].T.to_string())
    
    # 4. PCA Plot
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    for c in range(k):
        subset = customers[customers["cluster"] == c]
        plt.scatter(X_pca[subset.index, 0], X_pca[subset.index, 1], s=15, alpha=0.5, label=segment_names[c])
    plt.legend(); plt.title("Customer Segments (PCA)"); plt.savefig("outputs/08_pca_visualization.png", dpi=150, bbox_inches="tight"); plt.close()
    
    return customers, profile