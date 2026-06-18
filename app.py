


import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InfluenceGuard AI",
    layout="wide",
    page_icon="📊"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_output.csv")

df = load_data()

# Add label column
df['iso_label'] = df['iso_pred'].map({0: 'Real', 1: 'Fake'})

# ---------------- TITLE ----------------
st.title("📊 InfluenceGuard AI Dashboard")
st.markdown("### 🚀 AI-powered Influencer Analytics Platform")

st.markdown("""
Detect suspicious influencers, analyze engagement behavior,
and rank creators using AI.
""")

st.divider()

# ---------------- KPI (FULL DATA — NO FILTER) ----------------
total_users = len(df)
fake_users = df["iso_pred"].sum()
real_users = total_users - fake_users
avg_score = df["influence_score"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Influencers", total_users)
col2.metric("Real Accounts", real_users)
col3.metric("Fake Accounts", fake_users)
col4.metric("Avg Influence Score", f"{avg_score:.2f}")

st.divider()

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.title("🔍 Filters")

platform_filter = st.sidebar.multiselect(
    "Select Platform",
    options=df["platform"].unique(),
    default=df["platform"].unique()
)

tier_filter = st.sidebar.multiselect(
    "Select Tier",
    options=df["tier"].unique(),
    default=df["tier"].unique()
)

filtered_df = df[
    (df["platform"].isin(platform_filter)) &
    (df["tier"].isin(tier_filter))
]

# ---------------- TOP INFLUENCERS ----------------
st.subheader("🏆 Top Influencers")

top_df = filtered_df.nlargest(10, "influence_score")

fig_top = px.bar(
    top_df,
    x="influence_score",
    y="username",
    color="platform",
    orientation="h"
)

st.plotly_chart(fig_top, use_container_width=True)

# ---------------- FAKE VS REAL ----------------
col5, col6 = st.columns(2)

with col5:
    st.subheader("🛑 Fake vs Real")

    fig_pie = px.pie(
        filtered_df,
        names="iso_label"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with col6:
    st.subheader("📊 Tier Distribution")

    fig_tier = px.histogram(
        filtered_df,
        x="tier",
        color="tier"
    )

    st.plotly_chart(fig_tier, use_container_width=True)

st.divider()

# ---------------- ENGAGEMENT ----------------
st.subheader("📉 Engagement Analysis")

fig_scatter = px.scatter(
    filtered_df,
    x="followers",
    y="engagement_rate",
    color="platform",
    size="influence_score",
    hover_data=["username"]
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------- MODEL EXPLAINABILITY ----------------
st.subheader("🧠 How Models Work")

st.markdown("""
### 🔹 Isolation Forest
- Detects anomalies by isolating data points  
- Fake users behave differently → easier to isolate  
- More negative score = more suspicious  

### 🔹 K-Means
- Groups users based on behavior  
- Finds clusters of similar engagement patterns  
""")

# ---------------- ANOMALY SCORE ----------------
st.subheader("📊 Anomaly Score Distribution")

fig_iso = px.histogram(
    df,
    x="iso_score",
    color="iso_label"
)

st.plotly_chart(fig_iso, use_container_width=True)

# ---------------- CLUSTERS ----------------
st.subheader("📊 K-Means Clusters")

fig_cluster = px.scatter(
    filtered_df,
    x="engagement_rate",
    y="like_ratio",
    color="cluster",
    hover_data=["username"]
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ---------------- CLUSTER TABLE ----------------
st.subheader("📋 Cluster Behavior")

cluster_table = df.groupby("cluster")[
    ["engagement_rate", "like_ratio", "share_ratio"]
].mean().round(3)

st.dataframe(cluster_table)

st.divider()

# ---------------- SAMPLE USERS ----------------
st.subheader("👤 Sample Users per Tier")

sample_df = df.groupby("tier").head(3)

st.dataframe(
    sample_df[
        ["username", "tier", "influence_score", "iso_label"]
    ]
)

# ---------------- SUSPICIOUS USERS ----------------
st.subheader("🚨 Suspicious Influencers")

suspicious = df[df["iso_pred"] == 1]

st.dataframe(
    suspicious[
        ["username", "platform", "followers", "influence_score"]
    ].head(20)
)

st.divider()

# ---------------- INSIGHTS ----------------
st.subheader("💡 Key Insights")

st.markdown("""
- High engagement does not always mean authenticity  
- Fake users show abnormal interaction patterns  
- Isolation Forest ranks suspicious users  
- K-Means identifies behavior groups  
""")

# ---------------- DOWNLOAD ----------------
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered Data",
    csv,
    "filtered_influencers.csv",
    "text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | InfluenceGuard AI")