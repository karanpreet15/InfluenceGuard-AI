# 📊 InfluenceGuard AI

### AI-Powered Influencer Analytics & Fake Detection System

---

## 🚀 Overview

**InfluenceGuard AI** is an end-to-end data analytics and machine learning system designed to identify fake or low-quality influencers and rank authentic influencers using behavioral insights.

The system leverages **unsupervised learning** techniques to detect anomalies in engagement patterns and provides a **business-ready dashboard** for marketing decision-making.

---

## 🎯 Problem Statement

Brands often struggle to identify:

* Fake influencers
* Inflated engagement metrics
* Low-quality audiences

Additionally, real-world datasets **lack reliable labels**, making supervised learning impractical.

---

## 🧠 Solution Approach

This project uses **unsupervised learning** to analyze influencer behavior:

* 🔹 **Isolation Forest** → Detects anomalous (suspicious) accounts
* 🔹 **K-Means Clustering** → Groups influencers based on behavior
* 🔹 **Influence Score Model** → Ranks influencers using weighted metrics

---

## ⚙️ Tech Stack

* **Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn
* **Visualization**: Matplotlib, Seaborn, Plotly
* **Dashboard**: Streamlit
* **Tools**: Jupyter Notebook, VS Code

---

## 📊 Key Features

* ✔️ Fake influencer detection using anomaly detection
* ✔️ Behavioral clustering of users
* ✔️ Custom Influence Score (0–100)
* ✔️ Interactive dashboard for business insights
* ✔️ Multi-platform dataset (Instagram, Twitter, Facebook - simulated)
* ✔️ Real-time filtering (platform, tier)

---

## 🧩 Machine Learning Workflow

1. **Data Collection**

   * Kaggle dataset + synthetic multi-platform data

2. **Data Preprocessing**

   * Handling missing values
   * Removing duplicates
   * Cleaning inconsistent data

3. **Feature Engineering**

   * Engagement rate
   * Like/comment/share ratios
   * Platform-adjusted metrics
   * Behavioral flags

4. **Modeling**

   * Isolation Forest (Anomaly Detection)
   * K-Means Clustering

5. **Evaluation** *(Validation only)*

   * Precision, Recall, F1 Score

6. **Influence Score Calculation**

   * Engagement (40%)
   * Authenticity (30%)
   * Platform (20%)
   * Viral Boost (10%)

7. **Visualization & Dashboard**

   * Plotly + Streamlit

---

## 📈 Dashboard Highlights

* 📊 KPI Metrics (Total, Real, Fake, Score)
* 🏆 Top Influencers
* 🛑 Fake vs Real Distribution
* 📉 Engagement Analysis
* 🧠 Model Explainability
* 📋 Cluster Behavior Analysis
* 👤 Sample Users
* 🚨 Suspicious Accounts

---

## 📂 Project Structure

```
InfluenceGuard_AI/
│
├── data/
│   └── final_output.csv
│
├── notebooks/
│   ├── Data_Collection.ipynb
│   ├── Data_preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── Feature_eng.ipynb
│   ├── models.ipynb
│   └── app.py
│

│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```
streamlit run notebooks/app.py
```

### 3️⃣ Open in browser

```
http://localhost:8501
```

---

## Limitations

*  Relies on unsupervised learning (no direct label optimization)
*  Synthetic data may not fully reflect real-world behavior
*  K-Means assumes simple cluster shapes
*  Not real-time (batch processing)

---

## 🚀 Future Enhancements

* 🔹 Integrate real-time social media APIs
* 🔹 Use semi-supervised learning
* 🔹 Apply graph-based fake network detection
* 🔹 Deploy as a live web application
* 🔹 Add explainable AI (XAI) features

---

## 💡 Key Insight

> “Engagement drives influence, but authenticity defines it.”

---

## 🙌 Conclusion

This project demonstrates how **unsupervised learning can solve real-world problems** where labeled data is unavailable, while still delivering **actionable business insights**.

---

## 📬 Contact

**Karanpreet Singh**
📧 [ksingh6_be22@thapar.edu](mailto:ksingh6_be22@thapar.edu)
📍 Mohali, Punjab

---

