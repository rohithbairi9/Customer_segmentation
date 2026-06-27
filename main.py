import sys
import os

# 1. Add the 'src' directory to the Python path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 2. THEN import your modules
from data_processing import load_and_clean_data
from feature_engineering import build_customer_features
from model_training import preprocess_and_scale, find_optimal_k, train_final_model
from clv_prediction import train_clv_model

def main():
    print("="*70)
    print("STARTING ADVANCED CUSTOMER SEGMENTATION PIPELINE")
    print("="*70)

    # 1. Load & Clean
    df = load_and_clean_data("data/Online Retail.csv")

    # 2. Feature Engineering
    customers, feature_cols, skewed_cols = build_customer_features(df)

    # 3. Preprocessing
    X_scaled, scaler = preprocess_and_scale(customers, feature_cols, skewed_cols)

    # 4. Find K & Train Clusters (K-Means + GMM)
    best_k = find_optimal_k(X_scaled)
    OPTIMAL_K = 4 
    final_customers, profile = train_final_model(X_scaled, customers, feature_cols, k=OPTIMAL_K)

    # 5. Predict Customer Lifetime Value (CLV)
    final_customers, clv_importance = train_clv_model(final_customers)
    
    # Save CLV feature importance
    clv_importance.to_csv("outputs/clv_feature_importance.csv", index=False)

    # 6. Save Final Output (Now contains RFM + Clusters + GMM Probs + CLV!)
    final_customers.to_csv("outputs/segmented_customers.csv", index=False)
    profile.to_csv("outputs/segment_summary.csv")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE. Outputs saved.")
    print("="*70)

if __name__ == "__main__":
    main()