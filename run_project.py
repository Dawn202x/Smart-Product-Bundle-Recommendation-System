"""
Master Execution Pipeline for Smart Product Bundle Recommendation System
Executes:
Step 1: Data Collection / Generation (DMart Retail Transactions)
Step 2: Data Cleaning (Pandas, Handling Nulls, Duplicates, Profit Margins)
Step 3: Exploratory Data Analysis (EDA Charts & Financial Summaries)
Step 4: Market Basket Analysis (Apriori Algorithm & Association Rules)
Step 5: Profit-Aware Recommendation Verification
Step 6: Launch Streamlit Web Application
"""

import os
import sys
import subprocess

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("="*75)
    print("🚀 SMART PRODUCT BUNDLE RECOMMENDATION SYSTEM - END-TO-END PIPELINE")
    print("   Addressing: Co-purchase Affinity (Apriori) + Business Profit Margins")
    print("="*75)
    
    # Step 1: Data Generation
    print("\n[STEP 1/6] Running Data Collection & Generation...")
    from src.data_generator import generate_dmart_transactions
    generate_dmart_transactions()
    
    # Step 2: Data Cleaning
    print("\n[STEP 2/6] Running Data Cleaning & Margin Engineering...")
    from src.data_cleaning import clean_dmart_data
    clean_dmart_data()
    
    # Step 3: EDA
    print("\n[STEP 3/6] Running Exploratory Data Analysis & Visualizations...")
    from src.eda_analysis import perform_eda
    perform_eda()
    
    # Step 4: Apriori Market Basket Analysis
    print("\n[STEP 4/6] Running Market Basket Analysis (Apriori Engine)...")
    from src.apriori_engine import run_market_basket_analysis
    run_market_basket_analysis()
    
    # Step 5: Recommendation Engine Test
    print("\n[STEP 5/6] Verifying Profit-Aware Recommendation Engine...")
    from src.profit_aware_recommender import ProfitAwareRecommender
    rec = ProfitAwareRecommender()
    test_prod = "Intel Core i5 Thin Laptop"
    res = rec.recommend(test_prod, top_k=3)
    if res:
        print(f"✅ Primary Product: {test_prod}")
        print(f"📦 Recommended Smart Bundle: {[x['name'] for x in res['smart_bundle']['recommended_items']]}")
        print(f"💰 Bundle Price: ₹{res['smart_bundle']['bundle_price']:,.2f} | Bundle Net Profit: ₹{res['smart_bundle']['bundle_profit']:,.2f}")
        print(f"💡 {res['explanation']}")
        
    print("\n" + "="*75)
    print("🎉 ALL DATA & ML PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("="*75)
    print("\nTo launch the Streamlit Interactive Dashboard, run:")
    print("👉  streamlit run app.py")
    print("\nTo view the HTML web storefront, open:")
    print("👉  static/index.html")
    print("="*75)

if __name__ == "__main__":
    main()
