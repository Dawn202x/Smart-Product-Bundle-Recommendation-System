# 🛒 Smart Product Bundle Recommendation System (Profit-Aware Market Basket Analysis)

An end-to-end, multi-objective product bundle recommendation system built for modern retail (DMart) that combines **Market Basket Analysis (Apriori Algorithm)** with **Unit Economics (Profit Margin Optimization)**.

---

## 🎯 Project Core Innovation (Addressing Sir's Feedback)
> **Problem:** Traditional Market Basket Analysis (Apriori) only recommends products that customers buy frequently together (e.g. Milk & Bread). However, staple items often carry razor-thin profit margins (5–12%). Recommending only high-sales/low-margin items misses huge profit opportunities for the business.
> 
> **Our Solution:** A **Dual-Objective Recommendation Engine** that analyzes both **co-purchase affinity (Support, Confidence, Lift)** and **retail profit margin %**. It intelligently recommends bundles that maximize customer basket value and retailer gross profit.

---

## 🏗️ Project Architecture & Pipeline

```text
Step 1: Data Collection (DMart Retail Transactions with Cost & Selling Prices)
   ↓
Step 2: Data Cleaning (Pandas: Missing Values, Deduplication, Margin Engineering)
   ↓
Step 3: Exploratory Data Analysis (EDA: Volume vs Profit Margin BCG Matrix)
   ↓
Step 4: Market Basket Analysis (Apriori Engine: Support, Confidence, Lift)
   ↓
Step 5: Hybrid Profit-Aware Recommendation Engine (Multi-Objective Scoring)
   ↓
Step 6: Interactive Web Application (Streamlit Dashboard + SmartCart AI HTML Storefront)
```

---

## 🚀 Quick Start Guide

### 1. Run the Complete Data & Modeling Pipeline (Steps 1 to 5)
```bash
python run_project.py
```
This will automatically generate the dataset, clean the records, compute profit margins, produce EDA charts, generate 200+ association rules, and test the recommendation engine.

### 2. Launch the Streamlit Interactive Dashboard (Step 6)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
Features:
- **Interactive Bundle Recommender:** Pick any item (e.g., *Laptop*, *Bread*, *Smartphone*, *Shampoo*) and compare **Traditional Apriori vs Smart Profit-Aware** recommendations side-by-side.
- **Retail Analytics & EDA:** Real-time revenue, gross profit, and Volume vs Margin scatter matrix.
- **Association Rules Explorer:** Filter rules by Lift, Confidence, and Consequent Margin %.
- **Viva Defense Walkthrough:** In-depth guide with answers to examiner questions.

### 3. Open the SmartCart AI Storefront (HTML Web App)
You can directly open `static/index.html` in your browser (Google Chrome / Edge) to experience the responsive shopping storefront with smart combos and AI match scoring.

---

## 📊 Mathematical Formulation

### 1. Traditional Association Rule Mining
- **Support:** $P(A \cap B) = \frac{\text{Count}(A \cap B)}{N_{\text{total}}}$
- **Confidence:** $P(B \mid A) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}$
- **Lift:** $\frac{\text{Confidence}(A \to B)}{\text{Support}(B)}$

### 2. Profit Margin & Hybrid Multi-Objective Scoring
- **Profit Margin %:**
  $$\text{Margin}(B) = \frac{\text{Selling Price}(B) - \text{Cost Price}(B)}{\text{Selling Price}(B)} \times 100$$
- **Smart Bundle Score:**
  $$\text{Score}(A \to B) = \alpha \times \text{Norm}(\text{Lift}) + \beta \times \text{Norm}(\text{Margin}) + \gamma \times \text{Confidence}$$
  *(where $\alpha + \beta + \gamma = 1$, adjustable via sliders in Streamlit)*

---

## 📁 Repository Structure
```text
smart_bundle_rec/
│
├── data/
│   ├── raw_dmart_transactions.csv     # Raw generated retail transaction log
│   ├── cleaned_dmart_dataset.csv      # Cleaned data with Profit and Margin %
│   ├── association_rules.csv          # Enriched association rules with unit economics
│   ├── eda_product_summary.csv        # Product performance summary
│   └── eda_charts/                    # High-res charts (Volume vs Margin Matrix)
│
├── src/
│   ├── data_generator.py              # DMart retail transaction generator
│   ├── data_cleaning.py               # Data cleaning & financial engineering pipeline
│   ├── eda_analysis.py                # EDA and visualization script
│   ├── apriori_engine.py              # Pure Python Apriori & Association rules miner
│   └── profit_aware_recommender.py    # Dual-objective bundle recommender
│
├── static/
│   └── index.html                     # SmartCart AI Interactive Frontend
│
├── app.py                             # Streamlit Web Application
├── run_project.py                     # One-click master pipeline execution script
├── requirements.txt                   # Project dependencies
├── PROJECT_REPORT.md                  # Comprehensive viva documentation
└── README.md                          # Quick start instructions
```
