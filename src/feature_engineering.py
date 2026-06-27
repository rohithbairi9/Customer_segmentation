import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_customer_features(df):
    """Aggregates transactions to customer-level RFM + extended + NLP features."""
    print("Engineering customer-level features...")
    
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Core RFM
    rfm = df.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
    ).reset_index()

    # Extended
    extended = df.groupby("CustomerID").agg(
        avg_order_value=("TotalPrice", "mean"),
        std_order_value=("TotalPrice", "std"),
        total_items=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        avg_items_per_order=("Quantity", "mean"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        avg_days_between_orders=("InvoiceDate", lambda x: (x.max() - x.min()).days / (x.nunique() - 1) if x.nunique() > 1 else 0),
        country=("Country", "first"),
    ).reset_index()

    customers = rfm.merge(extended, on="CustomerID")

    # Derived Features
    customers["tenure_days"] = (reference_date - customers["first_purchase"]).dt.days
    customers["tenure_months"] = customers["tenure_days"] / 30.44
    customers["basket_diversity"] = customers["unique_products"] / customers["frequency"]
    customers["spend_per_month"] = customers["monetary"] / customers["tenure_months"].replace(0, 1)
    customers["order_value_stability"] = (customers["std_order_value"] / customers["avg_order_value"]).fillna(0)
    customers["is_uk"] = (customers["country"] == "United Kingdom").astype(int)

    customers = customers.drop(columns=["first_purchase", "last_purchase", "country"]).fillna(0)

    # --- TIME-SERIES FEATURES ---
    print("Calculating time-series trends...")
    customers = _add_time_series_features(df, customers)

    # --- NLP TEXT FEATURES ---
    print("Extracting NLP purchase topics...")
    customers = _add_nlp_features(df, customers)

    # Define columns for later steps (Now 19 features!)
    feature_cols = [
        "recency", "frequency", "monetary", "avg_order_value", "std_order_value",
        "total_items", "unique_products", "avg_items_per_order", "avg_days_between_orders",
        "tenure_days", "basket_diversity", "spend_per_month", "order_value_stability", "is_uk",
        "spend_trend_slope", "monthly_spend_std",
        "nlp_topic_1", "nlp_topic_2", "nlp_topic_3"  # <-- NEW NLP Features
    ]
    
    skewed_cols = [
        "frequency", "monetary", "total_items", "unique_products", 
        "avg_items_per_order", "spend_per_month", "std_order_value"
    ]

    print(f"Created {len(customers):,} customer profiles with {len(feature_cols)} features.")
    return customers, feature_cols, skewed_cols


def _add_time_series_features(df, customers):
    """Calculates if a customer's spend is trending up/down over time."""
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_spend = df.groupby(['CustomerID', 'InvoiceMonth'])['TotalPrice'].sum().reset_index()
    
    trends = []
    for cust_id in customers['CustomerID']:
        cust_data = monthly_spend[monthly_spend['CustomerID'] == cust_id].sort_values('InvoiceMonth')
        
        if len(cust_data) >= 2:
            x = range(len(cust_data))
            y = cust_data['TotalPrice'].values
            slope, _, _, _, _ = linregress(x, y)
            monthly_std = np.std(y)
        else:
            slope = 0.0
            monthly_std = 0.0
            
        trends.append({'CustomerID': cust_id, 'spend_trend_slope': slope, 'monthly_spend_std': monthly_std})
        
    return customers.merge(pd.DataFrame(trends), on='CustomerID')


def _add_nlp_features(df, customers):
    """Uses NLP (TF-IDF + SVD) to extract 3 latent topics from purchase descriptions."""
    # Group all descriptions bought by a single customer into one big text block
    df["Description"] = df["Description"].fillna("").astype(str)
    customer_texts = df.groupby("CustomerID")["Description"].apply(lambda x: " ".join(x)).reset_index()
    
    # Convert text to a TF-IDF matrix (measures word importance)
    # max_features=500 keeps it lightweight and fast
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(customer_texts["Description"])
    
    # Reduce the 500 words down to 3 "Topics" using SVD (Latent Semantic Analysis)
    svd = TruncatedSVD(n_components=3, random_state=42)
    topic_matrix = svd.fit_transform(tfidf_matrix)
    
    # Create dataframe for the 3 topics
    topic_df = pd.DataFrame(topic_matrix, columns=["nlp_topic_1", "nlp_topic_2", "nlp_topic_3"])
    topic_df["CustomerID"] = customer_texts["CustomerID"]
    
    return customers.merge(topic_df, on="CustomerID")