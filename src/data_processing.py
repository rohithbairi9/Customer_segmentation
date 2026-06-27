import pandas as pd
import numpy as np

def load_and_clean_data(filepath="data/Online Retail.csv"):
    """Loads raw CSV and performs deep cleaning."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, encoding="latin1")
    print(f"Raw rows: {len(df):,}")

    # Drop nulls
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    
    # Remove cancelled orders
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    
    # Remove negatives
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    
    # Dates & Total Price
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    
    # Remove outliers (1st and 99th percentile)
    for col in ["Quantity", "UnitPrice", "TotalPrice"]:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    print(f"Clean rows: {len(df):,} | Unique Customers: {df['CustomerID'].nunique():,}")
    return df