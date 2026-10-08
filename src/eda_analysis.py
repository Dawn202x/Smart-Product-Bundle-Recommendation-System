"""
Step 3: Exploratory Data Analysis (EDA)
Generates:
1. Top Selling Products (Frequency & Volume)
2. Most Profitable Products (Total Profit Generated)
3. Sales Volume vs. Profit Margin Matrix (BCG / Quadrant Matrix)
4. Category-level Revenue & Profit Margin breakdown
5. Exportable visualization figures saved to 'data/eda_charts/'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def perform_eda(input_path="data/cleaned_dmart_dataset.csv", output_dir="data/eda_charts"):
    print("="*60)
    print("📈 STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Clean data first.")
        
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary statistics
    total_txns = df["InvoiceNo"].nunique()
    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    avg_order_value = total_revenue / total_txns
    overall_margin = (total_profit / total_revenue) * 100
    
    print(f"📊 Total Transactions (Invoices): {total_txns:,}")
    print(f"💰 Total Store Revenue: ₹{total_revenue:,.2f}")
    print(f"💵 Total Gross Profit: ₹{total_profit:,.2f} ({overall_margin:.1f}% Overall Margin)")
    print(f"🛒 Average Order Value (AOV): ₹{avg_order_value:,.2f}")
    
    # 1. Product-level Aggregations
    prod_summary = df.groupby(["ProductName", "Category"]).agg(
        Total_Quantity=("Quantity", "sum"),
        Frequency=("InvoiceNo", "count"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum"),
        Unit_Price=("SellingPrice", "mean"),
        Cost_Price=("CostPrice", "mean"),
        Profit_Margin_Pct=("ProfitMarginPct", "mean")
    ).reset_index()
    
    # 2. Chart 1: Top 10 Most Frequently Purchased Items (Volume Drivers)
    top_volume = prod_summary.sort_values(by="Total_Quantity", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    bars = plt.barh(top_volume["ProductName"][::-1], top_volume["Total_Quantity"][::-1], color="#7b1fa2")
    plt.title("Top 10 Most Frequently Purchased Products (Volume Drivers)", fontsize=13, fontweight="bold")
    plt.xlabel("Total Units Sold")
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "top_selling_products.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved Chart: {chart1_path}")
    
    # 3. Chart 2: Top 10 Products by Total Profit Generated (Margin Drivers)
    top_profit = prod_summary.sort_values(by="Total_Profit", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_profit["ProductName"][::-1], top_profit["Total_Profit"][::-1], color="#16803c")
    plt.title("Top 10 Most Profitable Products (Total Profit Generated ₹)", fontsize=13, fontweight="bold")
    plt.xlabel("Total Profit (₹)")
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "top_profitable_products.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved Chart: {chart2_path}")
    
    # 4. Chart 3: Retail Strategy Matrix: Sales Volume vs Profit Margin %
    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(
        prod_summary["Total_Quantity"],
        prod_summary["Profit_Margin_Pct"],
        s=prod_summary["Total_Profit"] / 80,
        c=prod_summary["Profit_Margin_Pct"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black"
    )
    plt.colorbar(scatter, label="Profit Margin %")
    
    # Reference median lines for 4 quadrants
    median_vol = prod_summary["Total_Quantity"].median()
    median_margin = prod_summary["Profit_Margin_Pct"].median()
    plt.axvline(median_vol, color="grey", linestyle="--", alpha=0.6)
    plt.axhline(median_margin, color="grey", linestyle="--", alpha=0.6)
    
    # Annotate notable points
    for _, row in prod_summary.iterrows():
        if row["Total_Quantity"] > median_vol * 1.3 or row["Profit_Margin_Pct"] > 45:
            plt.annotate(
                row["ProductName"][:15], 
                (row["Total_Quantity"], row["Profit_Margin_Pct"]),
                fontsize=8,
                xytext=(5, 4), 
                textcoords="offset points"
            )
            
    plt.text(median_vol*1.4, median_margin*1.8, "[STAR] HIGH VOLUME + HIGH MARGIN\n(Star Bundles)", color="#16803c", fontweight="bold", fontsize=10)
    plt.text(median_vol*1.4, median_margin*0.3, "[STAPLE] HIGH VOLUME + LOW MARGIN\n(Traffic Drivers / Staples)", color="#d97706", fontweight="bold", fontsize=10)
    plt.text(median_vol*0.1, median_margin*1.8, "[GEM] LOW VOLUME + HIGH MARGIN\n(High-Margin Companions)", color="#7b1fa2", fontweight="bold", fontsize=10)
    
    plt.title("Retail Strategy Matrix: Sales Volume vs. Profit Margin %\n(Dual-Objective Retail Optimization Matrix)", fontsize=12, fontweight="bold")
    plt.xlabel("Sales Volume (Units Sold)")
    plt.ylabel("Profit Margin (%)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "sales_vs_profit_margin_matrix.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved Chart: {chart3_path}")
    
    # 5. Category Breakdown Summary
    cat_summary = df.groupby("Category").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum"),
        Items_Sold=("Quantity", "sum")
    ).reset_index()
    cat_summary["Margin_Pct"] = ((cat_summary["Total_Profit"] / cat_summary["Total_Revenue"]) * 100).round(2)
    cat_summary = cat_summary.sort_values(by="Total_Revenue", ascending=False)
    
    print("\n📦 Category Level Financial Summary:")
    print(cat_summary.to_string(index=False))
    
    # Export summary tables to CSV
    summary_path = os.path.join("data", "eda_product_summary.csv")
    prod_summary.to_csv(summary_path, index=False)
    print(f"💾 Product summary exported to: {summary_path}")
    print("="*60)
    return prod_summary, cat_summary

if __name__ == "__main__":
    perform_eda()
