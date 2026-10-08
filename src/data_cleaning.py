"""
Step 2: Data Cleaning Pipeline for DMart Dataset
Handles:
- Missing value imputation / drop
- Duplicate removal
- Unit profit and profit margin percentage computation
- Category standardisation
- Export of cleaned dataset for EDA and Apriori Modeling
"""

import pandas as pd
import numpy as np
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def clean_dmart_data(input_path="data/raw_dmart_transactions.csv", output_path="data/cleaned_dmart_dataset.csv"):
    print("="*60)
    print("🧹 STEP 2: DATA CLEANING & MARGIN ENRICHMENT PIPELINE")
    print("="*60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found. Please run data_generator.py first.")
        
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"📊 Initial Raw Records: {initial_rows}")
    
    # 1. Missing Values Check
    null_counts = df.isnull().sum()
    print("\n🔍 Missing values per column before cleaning:")
    print(null_counts[null_counts > 0])
    
    # Drop rows with nulls in critical fields
    df.dropna(subset=["InvoiceNo", "ProductName", "SellingPrice", "CostPrice"], inplace=True)
    # Impute customer age with median if missing
    if "CustomerAge" in df.columns and df["CustomerAge"].isnull().any():
        median_age = df["CustomerAge"].median()
        df["CustomerAge"] = df["CustomerAge"].fillna(median_age).astype(int)
    
    # 2. Duplicate Records Removal
    duplicates_count = df.duplicated().sum()
    print(f"\n🔄 Found and removing {duplicates_count} duplicate rows...")
    df.drop_duplicates(inplace=True)
    
    # 3. Data Type Standardisation & Sanity Validation
    df["SellingPrice"] = df["SellingPrice"].astype(float)
    df["CostPrice"] = df["CostPrice"].astype(float)
    df["Quantity"] = df["Quantity"].astype(int)
    
    # Ensure no zero/negative prices
    df = df[(df["SellingPrice"] > 0) & (df["CostPrice"] > 0) & (df["Quantity"] > 0)]
    
    # 4. Financial & Profit Margin Engineering
    # Dual-Objective Analysis: Calculate both sales volume and profit margin metrics
    df["Revenue"] = (df["Quantity"] * df["SellingPrice"]).round(2)
    df["TotalCost"] = (df["Quantity"] * df["CostPrice"]).round(2)
    df["Profit"] = (df["Revenue"] - df["TotalCost"]).round(2)
    df["UnitProfit"] = (df["SellingPrice"] - df["CostPrice"]).round(2)
    df["ProfitMarginPct"] = (((df["SellingPrice"] - df["CostPrice"]) / df["SellingPrice"]) * 100).round(2)
    
    # Margin Classifications: High, Medium, Low
    def classify_margin(pct):
        if pct >= 35.0:
            return "High Margin (>35%)"
        elif pct >= 20.0:
            return "Medium Margin (20-35%)"
        else:
            return "Low Margin (<20%)"
            
    df["MarginCategory"] = df["ProfitMarginPct"].apply(classify_margin)
    
    final_rows = len(df)
    print(f"\n✨ Cleaned Records Count: {final_rows} (Removed {initial_rows - final_rows} redundant rows)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset successfully exported to: {output_path}")
    print("="*60)
    return df

if __name__ == "__main__":
    clean_dmart_data()
