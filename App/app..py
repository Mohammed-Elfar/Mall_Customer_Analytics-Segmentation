import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

st.title(" Mall Customer Analytics & Segmentation")

# ────────────────────────────────────────────────
# Load Artifacts 
# ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    BASE_DIR = Path(__file__).resolve().parent.parent  
    # App/app.py  → project root

    return (
        pd.read_csv(BASE_DIR / "data/mall_customers_with_clusters.csv"),
        joblib.load(BASE_DIR / "models/customer_segmentation_Final_Pipeline.joblib"),
        joblib.load(BASE_DIR / "models/cluster_names.joblib"),
        pd.read_csv(BASE_DIR / "data/cluster_profiles.csv", index_col=0)
    )

df, pipeline, cluster_names, cluster_profiles = load_artifacts()

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "1- Analytics & Insights",
    "2- Clusters Analysis",
    "3- Segment New Customer by his Info",
    "4- Data information & Summary"
])

# ────────────────────────────────────────────────
# TAB 0 ─ Analytics & Insights
# ────────────────────────────────────────────────
with tab0:
    st.header("Customer Analytics & Insights")

    # ── Scatter plot ──
    st.subheader("How does Spending Score change with Annual Income and Age?")
    fig_scatter = px.scatter(
        df, 
        x="Annual Income (k$)", 
        y="Spending Score (1-100)",
        color="Age", 
        size="Age", 
        hover_data=["Gender", "Age"],
        title="Annual Income vs Spending Score (colored by Age)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Final insights (combined & polished) ──
    st.markdown("### 📌 Key Insights from the Data")

    st.markdown("""
**1. The famous pattern everyone sees in the scatter plot**

- **Low Income + High Spending** (bottom-left)  
  → People earning **less than ~40k** but spending a lot (60–100)  
  → Mostly **very young** customers (20–35 years old)  
  → These are **impulse / fun buyers** — they love shopping even with limited budget

- **Middle Income + Medium Spending** (center)  
  → Earn **~40k–70k**, spend in the middle (40–60)  
  → Ages very mixed  
  → Classic **everyday / average mall customers** — the stable core group

- **High Income (80k–140k) splits into two opposite behaviors**  
  - **Younger to middle-aged** (25–40 years): high spending (60–100) → **rich and love shopping** (premium / VIP group)  
  - **Older** (40–70 years): very low spending (0–40) → **rich but very careful** (hard to convince)

**Important message**: Age is **the biggest difference** among rich customers — young rich people spend a lot, older rich people spend almost nothing.
""")

    # ── Gender insight ──
    st.subheader("Average Spending by Gender")
    gender_avg = df.groupby("Gender")["Spending Score (1-100)"].mean().round(1)
    gender_df = pd.DataFrame({
        "Gender": ["Female", "Male"],
        "Spending Score": gender_avg.values
    })

    fig_gender = px.bar(
        gender_df, x="Gender", y="Spending Score",
        text_auto=True, color="Gender",
        title="Average Spending Score by Gender"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

    st.info("""
**Women spend slightly more** on average  
→ Female: ~51–52  
→ Male: ~48–49  

Difference is small (~3 points), but consistent → good opportunity for fashion, beauty, accessories, and lifestyle campaigns aimed at women.
""")

    # ── Age group income ──
    st.subheader("Average Income by Age Group")
    bins = [18, 25, 35, 45, 55, 70]
    labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
    df_temp = df.copy()
    df_temp["Age Group"] = pd.cut(df_temp["Age"], bins=bins, labels=labels, include_lowest=True)

    age_income = (
        df_temp.groupby("Age Group", observed=True)["Annual Income (k$)"]
        .mean()
        .round(1)
        .reset_index()
    )

    fig_age = px.bar(
        age_income, x="Age Group", y="Annual Income (k$)",
        text_auto=True, color="Age Group",
        title="Average Annual Income by Age Group"
    )
    st.plotly_chart(fig_age, use_container_width=True)

    st.info("""
**People aged 35–44 earn the most** (~72k on average)  
→ This is the strongest income group — target them for higher-priced products.
""")

    # ── Final strong recommendations ──
    st.markdown("### Final Marketing Recommendations")

    st.markdown("""
| Priority | Budget % | Target Group                     | Why? (Key Numbers)                          | Strong Actions to Take                                      |
|----------|----------|----------------------------------|---------------------------------------------|-------------------------------------------------------------|
| ★★★★★    | 50–60%   | Young Rich Big Spenders          | Highest spending (82) + high income (86k)   | VIP offers, luxury brands, personalized deals, exclusive events, premium loyalty |
| ★★★★     | 25–30%   | Young Fun Spenders               | Very young, high spending (59), volume potential | Flash sales, TikTok/Instagram ads, trendy items, student discounts, limited editions |
| ★★★      | 10–15%   | Old Normal Shoppers              | Largest group (30%), stable but average     | Everyday promotions, bundle deals, senior discounts — keep them coming |
| ★★       | Minimal  | Rich Careful People              | High income but very low spending           | Only high-quality/exclusive messages — don't waste budget |
| ★        | Minimal  | Young Girl Shoppers              | High income but lowest spending (14)        | Very low effort — focus on quality only                     |
""")

    st.success("""
**Most powerful recommindations**  
**Put 75–90% of your marketing budget on young customers (20–40 years old)**  
→ They drive almost all high spending and future growth.  

**Older rich people** (high income but low spending) are very hard to change — spend minimal effort there.
""")
# ────────────────────────────────────────────────
# TAB 1 ─ Cluster Analysis 
# ────────────────────────────────────────────────

with tab1:
    st.header("📊 Clusters Analysis")
     
    with st.expander("Cluster Names After Segmentation Analysis"):
        st.write("""
0 → Old Normal Shoppers  
1 → Young Rich Big Spenders  
2 → Young Fun Spenders  
3 → Rich Careful People  
4 → Young Girl Shoppers
""")
                                                    
    # ── Cluster distribution ──
    st.subheader("Customer Distribution by Cluster")
    
    counts = df["cluster"].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(1)
    
    dist_df = pd.DataFrame({
        "Cluster": counts.index.astype(str),
        "Percentage": percentages.values,
        "Count": counts.values
    })
    
    fig_dist = px.bar(
        dist_df,
        x="Cluster",
        y="Percentage",
        text_auto=True,
        title="Customer Distribution by Cluster (%)",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_dist.update_layout(xaxis_title="Cluster", yaxis_title="Percentage (%)")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Cluster profiles ──
    st.subheader("Cluster Profiles")
    
    for col, title in [
        ("Avg_Income",   "Average Annual Income by Cluster"),
        ("Avg_Spending", "Average Spending Score by Cluster"),
        ("Avg_Age",      "Average Age by Cluster")
    ]:
        fig = px.bar(
            cluster_profiles.reset_index(),
            x="cluster",
            y=col,
            text_auto=True,
            title=title,
            color="cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── New section: Marketing Recommendations ──
    st.markdown("---")
    st.subheader("Marketing Strategy & Budget Allocation")
    
    st.info("Based on cluster size, spending power, income and age — here is the recommended marketing focus")
    
    # Priority table
    recommendations = """
| Priority | Budget Share | Cluster Name              | Avg Age | Avg Income | Avg Spending | Key Characteristics & Why Focus                     | Recommended Actions                                      |
|----------|--------------|---------------------------|---------|------------|--------------|-----------------------------------------------------|----------------------------------------------------------|
| ★★★★★    | **50–60%**   | Young Rich Big Spenders   | 32.7    | 86.5k      | 82.1         | Highest spending + high income = biggest profit     | VIP offers, luxury brands, personalized discounts, exclusive events, premium loyalty program |
| ★★★★     | **25–30%**   | Young Fun Spenders        | 25.8    | ~40k       | 58.8         | Very young, high spending, perfect for volume       | Flash sales, TikTok/Instagram ads, limited editions, student/young discounts |
| ★★★      | **10–15%**   | Old Normal Shoppers       | 54.7    | 47.3k      | 41.3         | Largest group — stable but average spending         | Everyday promotions, bundle deals, senior discounts, keep them loyal |
| ★★       | Minimal      | Rich Careful People       | ~41     | ~85k       | 27.6         | High income but very low spending — hard to activate| Only high-quality/exclusive messages — low effort       |
| ★        | Minimal      | Young Girl Shoppers       | 39.5    | 85.2k      | 14.0         | High income but almost no spending                  | Minimal budget — focus on quality only                   |
"""
    
    st.markdown(recommendations)

    # Summary box
    st.success("""
**Final Strategy Summary**

Focus **75–90% of your marketing budget** on **young customers** (Clusters 1 & 2)  
→ They drive almost all high spending and growth potential.

Older and careful high-income groups → keep minimal effort only.

Youth-focused campaigns (social media, trendy products, VIP experiences) will give the best return.
""")

# ────────────────────────────────────────────────
# TAB 2 ─ Segment New Customer
# ────────────────────────────────────────────────
with tab2:
    st.header("Segment a New Customer")

    st.write("Slide to choose values:")

    # Gender as selectbox (still needed if pipeline expects it)
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)

    # Sliders instead of number_input
    age = st.slider(
        "Age",
        min_value=15,
        max_value=80,
        value=30,
        step=1,
        format="%d years"
    )

    income = st.slider(
        "Annual Income (k$)",
        min_value=0,
        max_value=200,
        value=50,
        step=5,
        format="%d k$"
    )

    spending = st.slider(
        "Spending Score (1–100)",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        format="%d"
    )

    if st.button("Predict Segment", type="primary"):
        
        # Create input with exact column names and numeric types
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": float(age),
            "Annual Income (k$)": float(income),
            "Spending Score (1-100)": float(spending)
        }])

        # Force the same column order as during training
        expected_order = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
        input_df = input_df[expected_order]

        # Debug: show what is being sent
        with st.expander("Debug – Input sent to model"):
            st.dataframe(input_df)

        try:
            cluster_id = pipeline.predict(input_df)[0]
            name = cluster_names.get(cluster_id, "Unknown")
            st.success(f"**Predicted Segment:** {name} (cluster {cluster_id})")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Tip: Make sure column names, order, and types match exactly what the pipeline was trained on.")
# ────────────────────────────────────────────────
# TAB 3 ─ Data & Info
# ────────────────────────────────────────────────
with tab3:
    st.header("Data information & Summary")

    # ── Quick stats cards ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(df))
    
    with col2:
        st.metric("Number of Columns", len(df.columns))
    
    with col3:
        st.metric("Clusters Found", df["cluster"].nunique())
    
    with col4:
        st.metric("Missing Values", df.isna().sum().sum())

    st.markdown("---")

    # ── Column names & explanations ──
    st.subheader("Columns in the Dataset")

    columns_info = {
        "CustomerID": "Unique identifier for each customer (not used in modeling)",
        "Gender": "Customer gender (Female / Male)",
        "Age": "Age of the customer in years",
        "Annual Income (k$)": "Yearly income of the customer in thousands of dollars",
        "Spending Score (1-100)": "Score assigned by the mall based on spending behavior and frequency (1 = low spender, 100 = high spender)",
        "cluster": "Assigned cluster number (0 to 4)",
        "cluster_name": "Human-readable name of the customer segment",
    }

    for col, desc in columns_info.items():
        if col in df.columns:
            st.markdown(f"**{col}**  \n{desc}")

    st.markdown("---")

    # ── Numerical summary ──
    st.subheader("Numerical Summary Statistics")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    desc = df[numerical_cols].describe().round(2).T
    desc["range"] = (desc["max"] - desc["min"]).round(2)

    st.dataframe(desc.style.format("{:,.2f}"))

    with st.expander("What do these numbers mean?"):
        st.markdown("""
- **count**: number of non-missing values  
- **mean**: average value  
- **std**: standard deviation (how spread out the values are)  
- **min / 25% / 50% / 75% / max**: minimum, quartiles, median, and maximum  
- **range**: difference between max and min
""")

    # ── Extra useful info ──
    st.subheader("Quick Data Overview")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Data Types**")
        dtypes = pd.DataFrame(df.dtypes.rename("Type"))
        st.dataframe(dtypes)

    with col_right:
        st.markdown("**Missing Values per Column**")
        missing = pd.DataFrame(df.isna().sum().rename("Missing")).query("Missing > 0")
        if missing.empty:
            st.success("No missing values in the dataset ✓")
        else:
            st.dataframe(missing)

    st.markdown("---")

    st.caption("Dataset: Mall Customers | Model: K-Means (5 clusters) | Last update: January 2026")
