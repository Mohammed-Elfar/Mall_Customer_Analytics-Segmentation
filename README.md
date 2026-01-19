#  Mall Customer Analytics & Segmentation

## Project Overview
This project focuses on **segmenting mall customers into meaningful groups** using **unsupervised machine learning (K-Means clustering)**.  
The goal is to help **business and marketing teams** better understand customer behavior and make **data-driven decisions**.

The project covers the **full ML lifecycle**:
data understanding → analysis → modeling → interpretation → deployment.

**The final solutin was deployed using Streamlit, providing an interactive application with multiple tabs:**

link of the app : https://customers-analytics-and-segmentation.streamlit.app/

---

##  Business Objective
- Identify distinct customer segments based on **age, income, and spending behavior**
- Enable **targeted marketing strategies**
- Improve **customer engagement and revenue potential**

---

##  1. Data Understanding & Preparation
Before modeling, the dataset was carefully reviewed to ensure data quality:
- Checked for **duplicates** and **missing values**
- Verified **data types** for all features
- Reviewed **basic statistics** (mean, min, max, distributions)

✔ The data was clean and required no major corrections.

---

##  2. Exploratory Data Analysis (EDA)
EDA was performed to better understand customer behavior:
- **Univariate analysis** for each feature (Age, Gender, Annual Income, Spending Score)
- **Relationship analysis** to explore how age, income, and spending interact

This step confirmed that **natural customer groupings exist**, making clustering meaningful.

---

##  3. Modeling & Clustering
A **machine learning pipeline** was built to ensure clean and reproducible modeling:
- Data preprocessing and scaling
- K-Means clustering

To select the optimal number of clusters, the following techniques were used:
- **Inertia**
- **Elbow Method**
- **Knee (Kneedle) technique**

✔ The optimal number of clusters was **k = 5**, providing clear and interpretable segments.

---

##  4. Cluster Analysis & Interpretation
Each cluster was analyzed to understand:
- Customer count
- Average age
- Average income
- Average spending score
- Gender distribution

Clusters were given **business-friendly names**, making the results easy to understand and actionable for **non-technical stakeholders**.

---

##  5. Deployment with Streamlit
The final solution was deployed using **Streamlit**, providing an interactive application with multiple tabs:

- **Analytics & Insights** – EDA and behavioral patterns  
- **Clusters Analysis** – Segment profiles and business meaning  
- **Customer Input & Prediction** – Assign new customers to segments  
- **Data Information & Summary** – Dataset statistics and summaries  

This allows business users to explore insights and use the model **without writing code**.

---

##  Key Business Insight
**Age is the strongest driver of spending behavior.**

- **Young customers (20–40 years old)** represent the **highest value and growth potential**
- **Older high-income customers** tend to spend less and are harder to activate


📌 Marketing efforts should focus mainly on **younger customer segments**.

