import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🛒", layout="wide")
sns.set_style("whitegrid")

# ── Load Data ──
@st.cache_data
def load_data():
    return pd.read_csv("outputs/segmented_customers.csv")

df = load_data()

# Colors for segments
segments = ["Loyal High-Spenders", "New Customers", "Steady Mid-Value", "Churning At-Risk"]
colors = sns.color_palette("Set2", 4)
color_map = {seg: c for seg, c in zip(segments, colors)}

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.header("🛒 Segmentation Project")
    st.markdown("*UCI Online Retail Dataset*")
    st.markdown("---")
    
    st.subheader("Filters")
    selected_segments = st.multiselect(
        "Filter by Segment",
        options=segments,
        default=segments
    )
    
    min_spend = st.slider("Minimum Total Spend (£)", 0, int(df["monetary"].max()), 0)
    max_recency = st.slider("Maximum Days Since Purchase", 0, int(df["recency"].max()), int(df["recency"].max()))
    
    st.markdown("---")
    st.subheader("Database Stats")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Segments Created", 4)

# Apply filters
mask = (
    (df["segment"].isin(selected_segments)) &
    (df["monetary"] >= min_spend) &
    (df["recency"] <= max_recency)
)
filtered_df = df[mask].copy()

# ═══════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════
st.title("📊 Customer Segmentation Dashboard")
st.markdown(f"Showing **{len(filtered_df):,}** customers out of **{len(df):,}**")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Revenue", f"£{filtered_df['monetary'].sum():,.0f}")
col2.metric("Avg Spend/Customer", f"£{filtered_df['monetary'].mean():,.0f}")
col3.metric("Avg Frequency", f"{filtered_df['frequency'].mean():.1f} orders")
col4.metric("Avg Recency", f"{filtered_df['recency'].mean():.0f} days")

st.markdown("---")

# ── Row 1: Pie Charts ──
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Customer Count")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    counts = filtered_df["segment"].value_counts()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", 
            colors=[color_map[s] for s in counts.index], startangle=90, textprops={"fontsize": 9})
    st.pyplot(fig1); plt.close()

with col_b:
    st.subheader("Revenue Share")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    rev = filtered_df.groupby("segment")["monetary"].sum()
    ax2.pie(rev, labels=rev.index, autopct="%1.1f%%",
            colors=[color_map[s] for s in rev.index], startangle=90, textprops={"fontsize": 9})
    st.pyplot(fig2); plt.close()

st.markdown("---")

# ── Row 2: RFM Histograms ──
st.subheader("RFM Metric Distributions")
col_c, col_d, col_e = st.columns(3)

for col, feat, title in zip([col_c, col_d, col_e], 
                            ["recency", "frequency", "monetary"],
                            ["Recency (Days)", "Frequency (Orders)", "Monetary (£)"]):
    with col:
        fig, ax = plt.subplots(figsize=(5, 4))
        for seg in selected_segments:
            subset = filtered_df[filtered_df["segment"] == seg]
            ax.hist(subset[feat], bins=30, alpha=0.5, label=seg, color=color_map[seg], edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)
        st.pyplot(fig); plt.close()

st.markdown("---")

# ── Row 3: Comparison Table ──
st.subheader("📋 Segment Comparison Table")
summary = filtered_df.groupby("segment").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("recency", "mean"),
    Avg_Frequency=("frequency", "mean"),
    Avg_Monetary=("monetary", "mean"),
    Avg_Order_Value=("avg_order_value", "mean"),
).round(1)

summary["Revenue_Share_%"] = (
    summary["Avg_Monetary"] * summary["Customers"] / 
    (summary["Avg_Monetary"] * summary["Customers"]).sum() * 100
).round(1)

st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ── Row 4: Business Insights ──
st.subheader("💡 Actionable Business Insights")

tabs = st.tabs(segments)

insights = {
    "Loyal High-Spenders": [
        "**73% of total revenue** comes from this group. Protect them at all costs.",
        "Do NOT over-discount; they already buy at full price.",
        "Offer VIP perks, early access to new products, and dedicated support.",
        "Cross-sell new categories—they have high basket diversity."
    ],
    "New Customers": [
        "Recently acquired (low tenure), but spending heavily per month right now.",
        "Critical 90-day window: nurture them to build a purchasing habit.",
        "Send an onboarding email sequence highlighting popular items.",
        "Offer a 'second purchase' discount to boost frequency."
    ],
    "Steady Mid-Value": [
        "Long-tenured customers who buy occasionally, but spend very little per month.",
        "They have the highest potential for upselling.",
        "Introduce tiered loyalty rewards to incentivize higher spend.",
        "Showcase premium product bundles to increase Avg Order Value."
    ],
    "Churning At-Risk": [
        "High recency (232 days) means they haven't bought in ~8 months.",
        "Only contributing 3.9% of revenue—act fast before they are lost forever.",
        "Launch a win-back campaign: 'We miss you' + 20% off coupon.",
        "Survey them to find out why they left (competitor? bad experience?)."
    ]
}

for tab, seg in zip(tabs, segments):
    with tab:
        for tip in insights[seg]:
            st.markdown(f"- {tip}")

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, & Streamlit | K-Means Clustering on UCI Online Retail Data")