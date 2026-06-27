import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os

def train_clv_model(customers):
    """Trains a regression model to predict Customer Monetary Value (CLV proxy)."""
    print("\n" + "="*70)
    print("PHASE 5: CLV (CUSTOMER LIFETIME VALUE) PREDICTION")
    print("="*70)
    
    # We drop direct monetary features to prevent data leakage.
    # The model must learn value from BEHAVIOR, not just from knowing what they already spent.
    clv_features = [
        "recency", "frequency", "tenure_days", "unique_products", 
        "basket_diversity", "avg_days_between_orders",
        "nlp_topic_1", "nlp_topic_2", "nlp_topic_3",
        "spend_trend_slope"  # Trend is a massive predictor of future value!
    ]
    
    X = customers[clv_features]
    y = customers["monetary"]
    
    # 1. Train Random Forest
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    # 2. Cross-Validation (Proves model robustness)
    print("Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"Cross-Validated R² Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # 3. Fit on all data for final deployment
    rf.fit(X, y)
    
    # 4. Predict for all customers
    customers["predicted_clv"] = rf.predict(X).round(2)
    
    # 5. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/clv_model.pkl")
    
    # 6. Feature Importance (What drives customer value?)
    importance = pd.DataFrame({
        'Feature': clv_features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop Predictors of Customer Value:")
    print(importance.to_string(index=False))
    
    return customers, importance