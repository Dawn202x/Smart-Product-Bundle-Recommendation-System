# 📄 Comprehensive Project Report

# Smart Product Bundle Recommendation System Using Profit-Aware Market Basket Analysis

---

## 📌 Executive Summary / Abstract

In modern e-commerce and retail supermarkets like DMart, product recommendation systems play a pivotal role in boosting Average Order Value (AOV) and customer satisfaction. Traditional recommendation algorithms, such as standard **Market Basket Analysis (Apriori Algorithm)**, focus solely on transaction co-occurrence frequencies (Support, Confidence, and Lift). 

However, from a business perspective, standard Apriori possesses a major flaw: **it treats all products equally in monetary value**. In retail, products like milk, salt, or bread have high sales volumes but razor-thin profit margins (typically 5%–12%). Recommending exclusively high-frequency staples yields minimal incremental profit for the business. Conversely, high-margin companion products (like specialty peanut butter, GaN fast chargers, laptop backpacks, or hair conditioners with 35%–60% margins) are often overlooked because their standalone sales volume is lower.

To solve this challenge, this project implements a **Profit-Aware Dual-Objective Recommendation Engine**. By enriching transaction records with **Cost Price, Selling Price, Quantity, and Product Category**, the system evaluates both **co-purchase affinity (Lift/Confidence)** and **profit margin %**. The resulting hybrid optimization algorithm delivers smart product bundles that provide customers with attractive bundle discounts while maximizing the retailer's gross profit.

---

## 🎯 Problem Statement & "Sir's" Specific Suggestion

### The Core Dilemma:
1. **High Sales, Low Margin Products:** Some products are bought very frequently (e.g., Bread, Milk, Atta, Smartphone), but the retailer earns very little margin per unit.
2. **High Profit, Lower Sales Products:** Some accessories and specialty products yield substantial margins (e.g., Laptop Backpacks, Mechanical Keyboards, Fast Chargers, Organic Jams, Hair Conditioners), but customers do not buy them without an intelligent recommendation or bundle incentive.
3. **The Teacher's Guideline:**
   > *"Sir suggested that we should also check the profit margin of each product. Some products have high sales but low profit. Some products have high profit but fewer sales. We should analyze both sales and profit margin before recommending product bundles. The system should recommend products that are frequently bought together and also give more profit to the business."*

---

## 🏗️ End-to-End 6-Step Project Workflow

```text
[Step 1: Data Collection]
   DMart Retail Transactions Dataset (Cost Price, Selling Price, Categories)
          ↓
[Step 2: Data Cleaning & Margin Engineering]
   Handle Nulls, Remove Duplicates, Compute Revenue, Profit, Margin %
          ↓
[Step 3: Exploratory Data Analysis (EDA)]
   Volume Drivers vs. Profit Drivers Matrix (BCG Retail Quadrant)
          ↓
[Step 4: Market Basket Analysis (Apriori Algorithm)]
   Frequent Itemset Mining, Support, Confidence, Lift Calculation
          ↓
[Step 5: Hybrid Profit-Aware Recommendation Engine]
   Multi-Objective Optimization: Score = α(Lift) + β(Margin%) + γ(Confidence)
          ↓
[Step 6: Interactive Web Application]
   Streamlit Business Dashboard & SmartCart AI Storefront UI
```

---

## 🔹 Step 1 – Data Collection & Schema Design

The dataset represents authentic retail transaction baskets inspired by supermarkets (like DMart) and online electronics portals.

### Schema Attributes:
| Attribute Name | Data Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | String | Unique transaction / basket identifier |
| `Date` | Datetime | Timestamp of transaction |
| `CustomerID` | String | Unique customer identifier |
| `CustomerAge` | Integer | Customer demographic age |
| `ProductID` | String | Unique SKU identifier |
| `ProductName` | String | Name of the product |
| `Category` | String | Department (Electronics, Grocery & Staples, Breakfast & Dairy, etc.) |
| `Quantity` | Integer | Units purchased in the transaction |
| `CostPrice` | Float (₹) | Wholesale acquisition cost per unit to the retailer |
| `SellingPrice` | Float (₹) | Retail price charged to the consumer |

---

## 🔹 Step 2 – Data Cleaning & Margin Engineering

Using Python and **Pandas**, data hygiene and financial feature engineering are performed:

1. **Missing Value Imputation & Scrubbing:**
   - Transactions missing `InvoiceNo`, `ProductName`, or prices are purged.
   - Demographic fields (`CustomerAge`) are imputed with the median.
2. **Deduplication:**
   - Identified and eliminated duplicate transaction rows via `df.drop_duplicates()`.
3. **Financial Formulas Applied:**
   - $\text{Revenue} = \text{Quantity} \times \text{SellingPrice}$
   - $\text{TotalCost} = \text{Quantity} \times \text{CostPrice}$
   - $\text{Profit} = \text{Revenue} - \text{TotalCost}$
   - $\text{UnitProfit} = \text{SellingPrice} - \text{CostPrice}$
   - $\text{ProfitMarginPct} = \left(\frac{\text{SellingPrice} - \text{CostPrice}}{\text{SellingPrice}}\right) \times 100$

---

## 🔹 Step 3 – Exploratory Data Analysis (EDA)

The EDA stage uncovers customer buying patterns and tests the hypothesis that sales volume does not correlate directly with profitability:

### Key Findings:
- **Top Volume Products:** Premium Wheat Bread, Fresh Cow Milk, Chakki Fresh Atta, Creamy Butter, Tata Iodized Salt.
- **Top Profit Generators:** Laptops, 5G Smartphones, Laptop Backpacks, Mechanical Keyboards, Anti-Dandruff Shampoo, Peanut Butter.
- **The Retail Strategy Matrix (4 Quadrants):**
  1. **Star Bundles (High Volume + High Margin):** Fast GaN Chargers, Padded Laptop Backpacks.
  2. **Traffic Drivers / Staples (High Volume + Low Margin):** Milk, Salt, Atta, Bread (Margin: 8%–18%).
  3. **Hidden Profit Gems (Low Volume + High Margin):** Crunchy Peanut Butter, Mixed Fruit Jam, Hair Conditioner, Fabric Softener (Margin: 40%–55%).
  4. **Slow Movers (Low Volume + Low Margin):** Phased out or re-bundled.

---

## 🔹 Step 4 – Market Basket Analysis & Apriori Algorithm

Market Basket Analysis identifies association rules between items purchased in the same invoice.

### Core Metrics:
1. **Support ($A \to B$):** Frequency with which both item $A$ and item $B$ appear together in the total transaction set $T$:
   $$\text{Support}(A \to B) = \frac{\text{count}(A \cup B)}{|T|}$$
2. **Confidence ($A \to B$):** Conditional probability that a customer buys $B$, given they bought $A$:
   $$\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
3. **Lift ($A \to B$):** The ratio of observed co-occurrence to expected co-occurrence if items were independent:
   $$\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)}$$
   - $\text{Lift} = 1$: Independent items.
   - $\text{Lift} > 1$: Positive association (items complement each other).

---

## 🔹 Step 5 – The Profit-Aware Recommendation Engine

### Why Traditional Apriori Fails:
If item $A$ is **Premium Wheat Bread**:
- Rule 1: `Bread -> Fresh Cow Milk` (Confidence = 65.8%, Lift = 3.21, Margin = **9.38%**, Profit = **₹6.00**)
- Rule 2: `Bread -> Creamy Butter` (Confidence = 65.5%, Lift = 3.91, Margin = **19.23%**, Profit = **₹10.00**)
- Rule 3: `Bread -> Crunchy Peanut Butter` (Confidence = 62.8%, Lift = 4.77, Margin = **45.71%**, Profit = **₹80.00**)
- Rule 4: `Bread -> Mixed Fruit Jam` (Confidence = 61.1%, Lift = 4.78, Margin = **39.29%**, Profit = **₹55.00**)

Under Traditional Apriori (ranking by confidence/support), the system recommends **Milk + Butter**, earning only **₹16.00** total add-on profit!
Under our **Profit-Aware Recommender**, the system recommends **Peanut Butter + Jam**, earning **₹135.00** total add-on profit (**+743% profit increase!**) while maintaining over 60% purchase affinity.

### Mathematical Formulation of Hybrid Score:
For candidate companion item $B$ given antecedent $A$:
$$\text{Norm}(\text{Lift}) = \frac{\text{Lift} - \text{Lift}_{\min}}{\text{Lift}_{\max} - \text{Lift}_{\min}}$$
$$\text{Norm}(\text{Margin}) = \frac{\text{Margin}(B) - \text{Margin}_{\min}}{\text{Margin}_{\max} - \text{Margin}_{\min}}$$
$$\text{Smart Bundle Score} = \alpha \times \text{Norm}(\text{Lift}) + \beta \times \text{Norm}(\text{Margin}) + \gamma \times \text{Confidence}$$

Where:
- $\alpha = 0.40$ (Co-purchase affinity weight)
- $\beta = 0.40$ (Profit margin weight)
- $\gamma = 0.20$ (Transaction confidence weight)

---

## 🔹 Step 6 – Web Applications

Two interactive user interfaces are provided:
1. **Streamlit Web Application (`app.py`):**
   - Interactive dropdown to select any product.
   - Sliders to dynamically adjust weights $\alpha, \beta, \gamma$ and Bundle Discount %.
   - Side-by-side card comparison: Traditional Apriori vs Smart Profit-Aware.
   - Interactive EDA visualizations and association rule filtering.
2. **SmartCart AI Storefront (`static/index.html`):**
   - High-fidelity retail storefront featuring the Laptop Productivity Suite, Breakfast Gourmet combo, and Smartphone kit with instant cart calculations.

---

## 🎓 Viva Voce & Defense Master Guide

### Q1: "What is your project about in simple terms?"
> *"I developed a Smart Product Bundle Recommendation System that helps retail businesses like DMart increase both order value and net profit. While existing systems recommend products solely based on what people frequently buy together, my system analyzes both co-purchase frequency (using the Apriori algorithm) and product profit margins. This ensures that the suggested bundles are relevant to the shopper while maximizing profitability for the store."*

### Q2: "What was your faculty advisor's / Sir's suggestion, and how did you solve it?"
> *"Sir pointed out that high-selling products often have very low profit margins (like milk or bread), while higher-margin items have moderate sales. He suggested analyzing both sales volume and profit margin before recommending bundles. I solved this by implementing a dual-objective scoring formula that normalizes and combines Lift, Profit Margin %, and Confidence, allowing retailers to optimize profitability without sacrificing customer relevance."*

### Q3: "What is the difference between Support, Confidence, and Lift?"
> - **Support:** How frequently the itemset appears in all transactions ($P(A \cap B)$).
> - **Confidence:** How often item $B$ is bought when item $A$ is in the cart ($P(B \mid A)$).
> - **Lift:** How much more likely $B$ is bought when $A$ is bought compared to random chance. A Lift greater than 1 indicates a true complementary relationship.

### Q4: "Give a concrete example from your results where your model outperformed standard Apriori."
> *"When a customer selects **Premium Wheat Bread**, standard Apriori recommends Milk (9.4% margin, ₹6 profit) and Butter (19.2% margin, ₹10 profit), yielding ₹16 add-on profit. Our Profit-Aware engine identifies that Peanut Butter (45.7% margin, ₹80 profit) and Fruit Jam (39.3% margin, ₹55 profit) also have high confidence (>60%) and high lift (>4.7). By recommending them, the store earns ₹135 add-on profit per bundle—an increase of over 700%—while giving the customer a 5% combo discount."*

### Q5: "What tools and technologies did you use?"
> *"Python 3.11, Pandas for data cleaning and financial metric computation, NumPy, Matplotlib for exploratory visualization, Apriori algorithm for Market Basket Analysis, and Streamlit for the interactive web application."*
