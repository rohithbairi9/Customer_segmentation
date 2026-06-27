💎 Advanced Customer Segmentation & CLV Prediction

    An end-to-end advanced machine learning pipeline applying K-Means, Gaussian Mixture Models (Soft Clustering), NLP, Time-Series Analysis, and Predictive CLV Modeling to the UCI Online Retail dataset.

🚀 Advanced Features Beyond Standard Clustering

This project goes far beyond basic RFM clustering by engineering 19 advanced features:

    NLP Topic Modeling: Uses TF-IDF and TruncatedSVD on product descriptions to extract 3 latent "purchase topics" (e.g., Home Decor vs. Toys) per customer.
    Time-Series Trend Analysis: Calculates the linear regression spend_trend_slope for each customer to determine if their spending is growing or declining over time.
    GMM Soft Clustering: Instead of hard boundaries, uses Gaussian Mixture Models to output the probability (e.g., 85% Loyal, 15% Churning) a customer belongs to each segment.
    Predictive CLV Modeling: Trains a RandomForestRegressor (validated via 5-Fold CV, R² = 0.55) to predict future Customer Lifetime Value without leaking historical revenue data.

🏷️ Segments Identified
Segment	Size	Revenue Share	Key Trait
Loyal High-Spenders	~25%	~72%	High frequency, low recency
New Customers	~25%	~10%	Short tenure, decent initial spend
Steady Mid-Value	~25%	~14%	Long tenure, low frequency
Churning At-Risk	~25%	~4%	Very high recency (~210 days)
🛠️ Tech Stack

    Python, Pandas, NumPy, SciPy (Linear Regression for trends)
    Scikit-Learn (K-Means, GMM, RandomForest, PCA, TruncatedSVD, TfidfVectorizer)
    Matplotlib, Seaborn
    Streamlit (Interactive Dashboard with CLV scatter plots & GMM risk tables)
    Joblib (Model serialization)

🚀 How to Run

    1.Clone the repo and install dependencies:
    pip install -r requirements.txt

    2.Run the modular ML pipeline:
    python3 main.py

    3.Launch the interactive dashboard:
    streamlit run app.py

📁 Project Structure

customer_segmentation/
├── app.py                          # Streamlit interactive dashboard
├── main.py                         # Pipeline orchestrator
├── requirements.txt
├── src/                            # Modular Python package
│   ├── __init__.py
│   ├── data_processing.py          # Cleaning & outlier removal
│   ├── feature_engineering.py      # RFM, NLP, Time-Series features
│   ├── model_training.py           # K-Means & GMM soft clustering
│   └── clv_prediction.py           # Random Forest CLV prediction
├── data/
│   └── Online Retail.csv           # (Ignored by git)
├── models/
│   ├── kmeans_model.pkl            # (Ignored by git)
│   ├── gmm_model.pkl               # (Ignored by git)
│   ├── clv_model.pkl               # (Ignored by git)
│   └── scaler.pkl                  # (Ignored by git)
├── outputs/
│   ├── segmented_customers.csv     # Final data (with probabilities & CLV)
│   ├── clv_feature_importance.csv  # What drives customer value
│   ├── segment_summary.csv
│   └── *.png                       # Visualizations
└── tests/
    └── test_pipeline.py            # Unit tests for pipeline validation