import pandas as pd
import numpy as np

# ── Load the CSV file ──
# We use encoding="latin1" because the UCI CSV sometimes has special characters that default utf-8 can't read
df = pd.read_csv("data/Online Retail.csv", encoding="latin1")

print(f"✅ Data loaded successfully!")
print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\n")

# ── First 5 rows ──
print("FIRST 5 ROWS:")
print("="*70)
print(df.head())

# ── Data types & missing values ──
print("\nDATA TYPES & MISSING VALUES:")
print("="*70)
print(df.info())

# ── Basic stats ──
print("\nBASIC NUMERIC STATS:")
print("="*70)
print(df.describe().round(2))

# ============================================
# STEP 2: Deep Data Cleaning
# ============================================

print(f"\n{'='*70}")
print("STARTING DEEP CLEANING...")
print(f"{'='*70}")
print(f"Raw rows:                  {len(df):>10,}")

# ── 2a. Remove rows without CustomerID ──
# We cannot segment anonymous transactions
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
print(f"After dropping null IDs:   {len(df):>10,} (removed {541909 - len(df):,})")

# ── 2b. Remove cancelled orders ──
# In this dataset, cancelled invoices start with the letter 'C'
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print(f"After removing cancels:    {len(df):>10,}")

# ── 2c. Remove negative or zero quantities/prices ──
# These represent returns, adjustments, or data entry errors
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print(f"After removing neg/zero:   {len(df):>10,}")

# ── 2d. Convert dates and create TotalPrice ──
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ── 2e. Remove extreme outliers ──
# K-Means is highly sensitive to outliers. We cap at the 1st and 99th percentiles.
outlier_cols = ["Quantity", "UnitPrice", "TotalPrice"]
for col in outlier_cols:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    before = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"  - Capped {col}: removed {before - len(df):,} extreme values")

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"FINAL CLEAN DATASET")
print(f"{'='*70}")
print(f"Clean Rows:                {len(df):>10,}")
print(f"Unique Customers:          {df['CustomerID'].nunique():>10,}")
print(f"Unique Products:           {df['StockCode'].nunique():>10,}")
print(f"Date Range:         {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
print(f"Total Clean Revenue:       £{df['TotalPrice'].sum():>10,.2f}")
print(f"{'='*70}")

# ============================================
# STEP 3: Feature Engineering (RFM + Extended)
# ============================================

print(f"\n{'='*70}")
print("ENGINEERING CUSTOMER-LEVEL FEATURES...")
print(f"{'='*70}")

# Reference date: 1 day after the last transaction in the dataset
# This is used to calculate "how many days ago" they purchased
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# ── 3a. Core RFM Features ──
rfm = df.groupby("CustomerID").agg(
    recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    frequency=("InvoiceNo", "nunique"),
    monetary=("TotalPrice", "sum"),
).reset_index()

# ── 3b. Extended Behavioral Features ──
extended = df.groupby("CustomerID").agg(
    avg_order_value=("TotalPrice", "mean"),
    std_order_value=("TotalPrice", "std"),       # How varied is their spending?
    total_items=("Quantity", "sum"),
    unique_products=("StockCode", "nunique"),    # How diverse is their taste?
    avg_items_per_order=("Quantity", "mean"),
    first_purchase=("InvoiceDate", "min"),
    last_purchase=("InvoiceDate", "max"),
    avg_days_between_orders=("InvoiceDate", lambda x: (
        x.max() - x.min()
    ).days / (x.nunique() - 1) if x.nunique() > 1 else 0),
    country=("Country", "first"),
).reset_index()

# ── 3c. Merge RFM + Extended ──
customers = rfm.merge(extended, on="CustomerID")

# ── 3d. Derived Features (The secret sauce) ──
# How long they have been a customer
customers["tenure_days"] = (reference_date - customers["first_purchase"]).dt.days
customers["tenure_months"] = customers["tenure_days"] / 30.44

# How many different products do they buy per order?
customers["basket_diversity"] = customers["unique_products"] / customers["frequency"]

# How much do they spend per month on average?
customers["spend_per_month"] = customers["monetary"] / customers["tenure_months"].replace(0, 1)

# Is their spending consistent? (Coefficient of Variation: lower = more predictable)
customers["order_value_stability"] = (
    customers["std_order_value"] / customers["avg_order_value"]
).fillna(0)

# Are they from the UK? (Majority of this dataset is UK, good to flag)
customers["is_uk"] = (customers["country"] == "United Kingdom").astype(int)

# ── 3e. Drop helper columns we no longer need ──
customers = customers.drop(columns=["first_purchase", "last_purchase", "country"])

# ── 3f. Handle any remaining NaNs ──
customers = customers.fillna(0)

print(f"✅ Feature engineering complete!")
print(f"   Transformed {len(df):,} transactions into {len(customers):,} customer profiles.")
print(f"\n{'='*70}")
print("CUSTOMER DATASET PREVIEW:")
print(f"{'='*70}")
print(customers.head().round(2))

print(f"\n{'='*70}")
print("FEATURE STATISTICS (Original Scale):")
print(f"{'='*70}")
print(customers.describe().round(2).T)

# ============================================
# STEP 4: Preprocessing for Clustering
# ============================================
from sklearn.preprocessing import StandardScaler

# ── 4a. Define the exact features we will use ──
feature_cols = [
    "recency", "frequency", "monetary", "avg_order_value",
    "std_order_value", "total_items", "unique_products",
    "avg_items_per_order", "avg_days_between_orders",
    "tenure_days", "basket_diversity", "spend_per_month",
    "order_value_stability", "is_uk"
]

X = customers[feature_cols].copy()

# ── 4b. Handle infinities (just in case of division by zero) ──
X = X.replace([np.inf, -np.inf], 0)

# ── 4c. Log-Transform heavily skewed features ──
# E-commerce data is ALWAYS right-skewed. Log1p smooths this out.
skewed_cols = [
    "frequency", "monetary", "total_items", 
    "unique_products", "avg_items_per_order", 
    "spend_per_month", "std_order_value"
]

print("Applying log-transform to skewed features:")
for col in skewed_cols:
    X[col] = np.log1p(X[col])
    print(f"  ✅ {col:<25} | Skewness: {X[col].skew():.2f}")

# ── 4d. Scale all features to mean=0, std=1 ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n{'='*70}")
print("SCALING COMPLETE")
print(f"{'='*70}")
print(f"Matrix shape: {X_scaled.shape}")
print(f"Mean (should be ~0): {X_scaled.mean(axis=0).round(4)}")
print(f"Std  (should be ~1): {X_scaled.std(axis=0).round(4)}")
print(f"{'='*70}")

# ============================================
# STEP 5: Find Optimal K
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

K_range = range(2, 11)
inertias = []
silhouette_scores = []

print(f"{'='*70}")
print("EVALUATING K FROM 2 TO 10...")
print(f"{'='*70}")

for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"K={k:2d}  |  Inertia={km.inertia_:>12.1f}  |  Silhouette={sil:.4f}")

# ── Plot Elbow and Silhouette side by side ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
ax1.set_ylabel("Inertia (Within-cluster SSE)", fontsize=12)
ax1.set_title("Elbow Method", fontsize=14)
ax1.set_xticks(list(K_range))
ax1.grid(True, alpha=0.3)

# Silhouette Plot
ax2.plot(K_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
best_k = list(K_range)[np.argmax(silhouette_scores)]
ax2.axvline(x=best_k, color="green", linestyle="--", linewidth=2,
            label=f"Best K={best_k}")
ax2.set_xlabel("Number of Clusters (K)", fontsize=12)
ax2.set_ylabel("Silhouette Score", fontsize=12)
ax2.set_title("Silhouette Score", fontsize=14)
ax2.set_xticks(list(K_range))
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle("Optimal K Selection", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("outputs/04_optimal_k.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n{'='*70}")
print(f"✅ RECOMMENDED K = {best_k} (Highest Silhouette Score)")
print(f"{'='*70}")
print("💡 Check the 'outputs/' folder for the plot!")
print("💡 Look at the Elbow plot: where does the line bend?")
print("💡 Look at the Silhouette plot: the highest peak is our chosen K.")

# ============================================
# STEP 6: Train Final K-Means Model
# ============================================
import joblib

# We override the mathematical best to use the business-optimal K=4
OPTIMAL_K = 4

print(f"\n{'='*70}")
print(f"TRAINING FINAL K-MEANS MODEL (K={OPTIMAL_K})...")
print(f"{'='*70}")

final_model = KMeans(n_clusters=OPTIMAL_K, n_init=30, random_state=42, max_iter=500)
cluster_labels = final_model.fit_predict(X_scaled)

# Attach labels back to our customer dataframe
customers["cluster"] = cluster_labels

print("\nCLUSTER DISTRIBUTION:")
print(customers["cluster"].value_counts().sort_index())

# ── Save the model and scaler to disk ──
joblib.dump(final_model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(f"\n{'='*70}")
print("✅ MODEL SAVED SUCCESSFULLY")
print(f"{'='*70}")
print("→ models/kmeans_model.pkl")
print("→ models/scaler.pkl")

# ============================================
# STEP 7: Cluster Profiling & Visualization
# ============================================

# ── 7a. Profile the clusters using ORIGINAL (non-log, non-scaled) values ──
profile_raw = customers.groupby("cluster").agg(
    count=("CustomerID", "size"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    avg_order_value=("avg_order_value", "mean"),
    avg_unique_products=("unique_products", "mean"),
    avg_tenure_days=("tenure_days", "mean"),
    avg_spend_per_month=("spend_per_month", "mean"),
).round(2)

# Calculate revenue share
profile_raw["revenue_share_%"] = (
    profile_raw["avg_monetary"] * profile_raw["count"]
    / (profile_raw["avg_monetary"] * profile_raw["count"]).sum() * 100
).round(1)

print(f"{'='*80}")
print("CLUSTER PROFILES (Original Scale):")
print(f"{'='*80}")
print(profile_raw.T.to_string())

# ── 7b. Relative Heatmap (% of Grand Mean) ──
# This makes it incredibly easy to spot patterns (e.g., >150% means very high)
grand_mean = customers[feature_cols].mean()
profile_scaled = customers.groupby("cluster")[feature_cols].mean()
profile_pct = (profile_scaled / grand_mean * 100).round(0)

plt.figure(figsize=(14, 7))
sns.heatmap(profile_pct.T, annot=True, fmt=".0f", cmap="YlGnBu",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% of Average Customer"})
plt.title("Cluster Profile Heatmap (% of Grand Mean)", fontsize=14)
plt.ylabel("Feature")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("outputs/05_cluster_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7c. RFM Box Plots ──
rfm_cols = ["recency", "frequency", "monetary"]
rfm_labels = ["Recency (Days)", "Frequency (Orders)", "Monetary (£)"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (col, label) in enumerate(zip(rfm_cols, rfm_labels)):
    sns.boxplot(data=customers, x="cluster", y=col, ax=axes[i], hue="cluster", palette="Set2", legend=False)
    axes[i].set_title(label, fontsize=13)
    axes[i].set_xlabel("Cluster")

plt.suptitle("Core RFM Distributions by Cluster", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("outputs/06_rfm_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7d. Extended Features Box Plots ──
ext_cols = ["avg_order_value", "unique_products", "spend_per_month", "tenure_days"]
ext_labels = ["Avg Order Value (£)", "Unique Products", "Spend/Month (£)", "Tenure (Days)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (col, label) in enumerate(zip(ext_cols, ext_labels)):
    sns.boxplot(data=customers, x="cluster", y=col, ax=axes[i], hue="cluster", palette="Set2", legend=False)
    axes[i].set_title(label, fontsize=12)

plt.suptitle("Extended Feature Distributions by Cluster", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("outputs/07_extended_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Profile tables printed and plots saved to outputs/")

# ============================================
# STEP 8: Name Segments, PCA, & Final Output
# ============================================
from sklearn.decomposition import PCA

# ── 8a. Name the clusters based on our analysis ──
segment_names = {
    0: "Loyal High-Spenders",
    1: "New Customers",
    2: "Steady Mid-Value",
    3: "Churning At-Risk",
}
customers["segment"] = customers["cluster"].map(segment_names)

print(f"\n{'='*80}")
print("FINAL SEGMENT SUMMARY:")
print(f"{'='*80}")
print(profile_raw[["count", "avg_recency", "avg_frequency", "avg_monetary", "revenue_share_%"]].T.to_string())

# ── 8b. 2D PCA Visualization ──
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = sns.color_palette("Set2", OPTIMAL_K)

for c in range(OPTIMAL_K):
    subset = customers[customers["cluster"] == c]
    plt.scatter(X_pca[subset.index, 0], X_pca[subset.index, 1],
                s=20, alpha=0.5, color=colors[c],
                label=f"{segment_names[c]} (n={len(subset):,})")

# Plot centroids
centroids_pca = pca.transform(final_model.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            s=250, marker="X", c="black", edgecolors="yellow", linewidths=2, zorder=5)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
plt.title("Customer Segments — 2D PCA Projection", fontsize=14)
plt.legend(fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig("outputs/08_pca_visualization.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8c. Pie Charts (Count vs Revenue) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Count pie
seg_counts = customers["segment"].value_counts()
ax1.pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(seg_counts)), startangle=90, textprops={"fontsize": 9})
ax1.set_title("Customer Count by Segment", fontsize=13)

# Revenue pie
rev_shares = profile_raw.set_index(profile_raw.index)["revenue_share_%"]
rev_labels = [segment_names[i] for i in rev_shares.index]
ax2.pie(rev_shares, labels=rev_labels, autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(rev_shares)), startangle=90, textprops={"fontsize": 9})
ax2.set_title("Revenue Share by Segment", fontsize=13)

plt.suptitle("Segment Distribution Overview", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("outputs/09_segment_pies.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8d. Save Final CSV ──
output_cols = [
    "CustomerID", "cluster", "segment", "recency", "frequency", "monetary",
    "avg_order_value", "unique_products", "tenure_days", "spend_per_month"
]
customers[output_cols].to_csv("outputs/segmented_customers.csv", index=False)

print(f"\n{'='*80}")
print("✅ PROJECT PIPELINE COMPLETE!")
print(f"{'='*80}")
print(f"Final labeled data saved to: outputs/segmented_customers.csv")
print(f"Total visualizations generated: 6")
print(f"\n💡 Next Step: Build the Streamlit Dashboard (app.py) to present this!")

# ============================================
# RUN ONCE: Generate Missing Documentation CSVs
# ============================================

# 1. Save Segment Summary
summary_to_save = profile_raw.copy()
summary_to_save["segment"] = summary_to_save.index.map(segment_names)
summary_to_save.to_csv("outputs/segment_summary.csv")
print("✅ Saved outputs/segment_summary.csv")

# 2. Save Feature Definitions
feature_doc = pd.DataFrame({
    "feature": feature_cols,
    "description": [
        "Days since last purchase (lower = more recent)",
        "Number of unique orders placed",
        "Total revenue from customer (£)",
        "Average spend per order (£)",
        "Standard deviation of order values (£)",
        "Total quantity of items purchased",
        "Number of distinct products bought",
        "Average items per order",
        "Average days between consecutive orders",
        "Days since first purchase (customer age)",
        "Unique products per order (variety)",
        "Total spend divided by tenure months",
        "CV of order value (lower = more consistent)",
        "1 if customer is from UK, 0 otherwise",
    ],
    "used_log_transform": [False if c not in skewed_cols else True for c in feature_cols],
})
feature_doc.to_csv("outputs/feature_definitions.csv", index=False)
print("✅ Saved outputs/feature_definitions.csv")