# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Page config
# st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🛒", layout="wide")
# sns.set_style("whitegrid")

# # ── Load Data ──
# @st.cache_data
# def load_data():
#     return pd.read_csv("outputs/segmented_customers.csv")

# df = load_data()

# # Colors for segments
# segments = ["Loyal High-Spenders", "New Customers", "Steady Mid-Value", "Churning At-Risk"]
# colors = sns.color_palette("Set2", 4)
# color_map = {seg: c for seg, c in zip(segments, colors)}

# # ═══════════════════════════════════════════
# # SIDEBAR
# # ═══════════════════════════════════════════
# with st.sidebar:
#     st.header("🛒 Segmentation Project")
#     st.markdown("*UCI Online Retail Dataset*")
#     st.markdown("---")
    
#     st.subheader("Filters")
#     selected_segments = st.multiselect(
#         "Filter by Segment",
#         options=segments,
#         default=segments
#     )
    
#     min_spend = st.slider("Minimum Total Spend (£)", 0, int(df["monetary"].max()), 0)
#     max_recency = st.slider("Maximum Days Since Purchase", 0, int(df["recency"].max()), int(df["recency"].max()))
    
#     st.markdown("---")
#     st.subheader("Database Stats")
#     st.metric("Total Customers", f"{len(df):,}")
#     st.metric("Segments Created", 4)

# # Apply filters
# mask = (
#     (df["segment"].isin(selected_segments)) &
#     (df["monetary"] >= min_spend) &
#     (df["recency"] <= max_recency)
# )
# filtered_df = df[mask].copy()

# # ═══════════════════════════════════════════
# # MAIN PAGE
# # ═══════════════════════════════════════════
# st.title("📊 Customer Segmentation Dashboard")
# st.markdown(f"Showing **{len(filtered_df):,}** customers out of **{len(df):,}**")

# # KPIs
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Filtered Revenue", f"£{filtered_df['monetary'].sum():,.0f}")
# col2.metric("Avg Spend/Customer", f"£{filtered_df['monetary'].mean():,.0f}")
# col3.metric("Avg Frequency", f"{filtered_df['frequency'].mean():.1f} orders")
# col4.metric("Avg Recency", f"{filtered_df['recency'].mean():.0f} days")

# st.markdown("---")

# # ── Row 1: Pie Charts ──
# col_a, col_b = st.columns(2)

# with col_a:
#     st.subheader("Customer Count")
#     fig1, ax1 = plt.subplots(figsize=(6, 5))
#     counts = filtered_df["segment"].value_counts()
#     ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", 
#             colors=[color_map[s] for s in counts.index], startangle=90, textprops={"fontsize": 9})
#     st.pyplot(fig1); plt.close()

# with col_b:
#     st.subheader("Revenue Share")
#     fig2, ax2 = plt.subplots(figsize=(6, 5))
#     rev = filtered_df.groupby("segment")["monetary"].sum()
#     ax2.pie(rev, labels=rev.index, autopct="%1.1f%%",
#             colors=[color_map[s] for s in rev.index], startangle=90, textprops={"fontsize": 9})
#     st.pyplot(fig2); plt.close()

# st.markdown("---")

# # ── Row 2: RFM Histograms ──
# st.subheader("RFM Metric Distributions")
# col_c, col_d, col_e = st.columns(3)

# for col, feat, title in zip([col_c, col_d, col_e], 
#                             ["recency", "frequency", "monetary"],
#                             ["Recency (Days)", "Frequency (Orders)", "Monetary (£)"]):
#     with col:
#         fig, ax = plt.subplots(figsize=(5, 4))
#         for seg in selected_segments:
#             subset = filtered_df[filtered_df["segment"] == seg]
#             ax.hist(subset[feat], bins=30, alpha=0.5, label=seg, color=color_map[seg], edgecolor="white")
#         ax.set_title(title, fontsize=11)
#         ax.legend(fontsize=7)
#         st.pyplot(fig); plt.close()

# st.markdown("---")

# # ── Row 3: Comparison Table ──
# st.subheader("📋 Segment Comparison Table")
# summary = filtered_df.groupby("segment").agg(
#     Customers=("CustomerID", "count"),
#     Avg_Recency=("recency", "mean"),
#     Avg_Frequency=("frequency", "mean"),
#     Avg_Monetary=("monetary", "mean"),
#     Avg_Order_Value=("avg_order_value", "mean"),
# ).round(1)

# summary["Revenue_Share_%"] = (
#     summary["Avg_Monetary"] * summary["Customers"] / 
#     (summary["Avg_Monetary"] * summary["Customers"]).sum() * 100
# ).round(1)

# st.dataframe(summary, use_container_width=True)

# st.markdown("---")

# # ── Row 4: Business Insights ──
# st.subheader("💡 Actionable Business Insights")

# tabs = st.tabs(segments)

# insights = {
#     "Loyal High-Spenders": [
#         "**73% of total revenue** comes from this group. Protect them at all costs.",
#         "Do NOT over-discount; they already buy at full price.",
#         "Offer VIP perks, early access to new products, and dedicated support.",
#         "Cross-sell new categories—they have high basket diversity."
#     ],
#     "New Customers": [
#         "Recently acquired (low tenure), but spending heavily per month right now.",
#         "Critical 90-day window: nurture them to build a purchasing habit.",
#         "Send an onboarding email sequence highlighting popular items.",
#         "Offer a 'second purchase' discount to boost frequency."
#     ],
#     "Steady Mid-Value": [
#         "Long-tenured customers who buy occasionally, but spend very little per month.",
#         "They have the highest potential for upselling.",
#         "Introduce tiered loyalty rewards to incentivize higher spend.",
#         "Showcase premium product bundles to increase Avg Order Value."
#     ],
#     "Churning At-Risk": [
#         "High recency (232 days) means they haven't bought in ~8 months.",
#         "Only contributing 3.9% of revenue—act fast before they are lost forever.",
#         "Launch a win-back campaign: 'We miss you' + 20% off coupon.",
#         "Survey them to find out why they left (competitor? bad experience?)."
#     ]
# }

# for tab, seg in zip(tabs, segments):
#     with tab:
#         for tip in insights[seg]:
#             st.markdown(f"- {tip}")

# st.markdown("---")
# st.caption("Built with Python, Scikit-Learn, & Streamlit | K-Means Clustering on UCI Online Retail Data")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Advanced Customer Segmentation & CLV", page_icon="💎", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("outputs/segmented_customers.csv")

df = load_data()

# Dynamic segments and colors based on data
segments = sorted(df["segment"].unique())
colors = sns.color_palette("Set2", len(segments))
color_map = {seg: c for seg, c in zip(segments, colors)}

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.header("💎 Advanced Analytics")
    st.markdown("*19 Features | GMM Soft Clustering | CLV Prediction*")
    st.markdown("---")
    
    st.subheader("Filters")
    selected_segments = st.multiselect("Filter by Segment", options=segments, default=segments)
    min_clv = st.slider("Minimum Predicted CLV (£)", 0, int(df["predicted_clv"].max()), 0)
    
    st.markdown("---")
    st.subheader("Database Stats")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Features Used", "19")
    st.metric("CLV R² Score", "0.553")

mask = (df["segment"].isin(selected_segments)) & (df["predicted_clv"] >= min_clv)
filtered_df = df[mask].copy()

# ═══════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════
st.title("📊 Advanced Customer Segmentation & CLV Dashboard")
st.markdown(f"Showing **{len(filtered_df):,}** customers")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predicted CLV", f"£{filtered_df['predicted_clv'].sum():,.0f}")
col2.metric("Avg Predicted CLV", f"£{filtered_df['predicted_clv'].mean():,.0f}")
col3.metric("Avg Actual Spend", f"£{filtered_df['monetary'].mean():,.0f}")
col4.metric("Avg Trend Slope", f"{filtered_df['spend_trend_slope'].mean():.1f}")

st.markdown("---")

# ── Row 1: Pie Charts ──
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Customer Count")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    counts = filtered_df["segment"].value_counts()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[color_map[s] for s in counts.index], startangle=90, textprops={"fontsize": 9})
    st.pyplot(fig1); plt.close()

with col_b:
    st.subheader("Predicted CLV Share")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    clv_share = filtered_df.groupby("segment")["predicted_clv"].sum()
    ax2.pie(clv_share, labels=clv_share.index, autopct="%1.1f%%", colors=[color_map[s] for s in clv_share.index], startangle=90, textprops={"fontsize": 9})
    st.pyplot(fig2); plt.close()

st.markdown("---")

# ── Row 2: CLV vs Actual Scatter Plot ──
st.subheader("Actual Spend vs. Predicted CLV")
fig3, ax3 = plt.subplots(figsize=(10, 6))
for seg in selected_segments:
    subset = filtered_df[filtered_df["segment"] == seg]
    ax3.scatter(subset["monetary"], subset["predicted_clv"], alpha=0.4, s=15, label=seg, color=color_map[seg])
ax3.plot([0, filtered_df["monetary"].max()], [0, filtered_df["monetary"].max()], 'r--', label="Perfect Prediction")
ax3.set_xlabel("Actual Historical Spend (£)")
ax3.set_ylabel("Predicted Future CLV (£)")
ax3.legend()
st.pyplot(fig3); plt.close()

st.markdown("---")

# ── Row 3: GMM Probabilities Insight ──
st.subheader("🧠 Soft Clustering Insights (GMM Probabilities)")
st.markdown("Below are customers who are primarily 'Loyal High-Spenders' but have a >10% probability of churning. **Intervention required!**")

prob_cols = [c for c in df.columns if c.startswith("prob_")]
if "prob_Churning At-Risk" in prob_cols and "prob_Loyal High-Spenders" in prob_cols:
    at_risk_vips = filtered_df[
        (filtered_df["prob_Loyal High-Spenders"] > 0.5) & 
        (filtered_df["prob_Churning At-Risk"] > 0.10)
    ].sort_values("prob_Churning At-Risk", ascending=False).head(10)

    # Find customers who are primarily Loyal, but have the HIGHEST relative churn risk
    # loyal_vips = filtered_df[filtered_df["prob_Loyal High-Spenders"] > 0.5]
    # at_risk_vips = loyal_vips.sort_values("prob_Churning At-Risk", ascending=False).head(10)
    
    if len(at_risk_vips) > 0:
        display_cols = ["CustomerID", "monetary", "predicted_clv", "spend_trend_slope", "prob_Loyal High-Spenders", "prob_Churning At-Risk"]
        st.dataframe(at_risk_vips[display_cols].round(3), use_container_width=True)
    else:
        st.info("No high-risk VIPs found in the current filter.")
else:
    st.warning("GMM probability columns not found in dataset.")

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, GMM, NLP & Streamlit | Advanced ML Pipeline")