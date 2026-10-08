"""
DMart Retail Transaction Dataset Generator
Generates realistic retail & supermarket transactions with:
- InvoiceID, Date, Customer_Age, Product_ID, Product_Name, Category, Cost_Price, Selling_Price, Quantity
- Realistic cross-category basket co-occurrences (Staples, Breakfast, Electronics, Personal Care)
- Dual distribution: High-sales/low-margin items vs. High-margin/medium-sales items
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import sys

# Configure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Product Catalog with accurate Indian Retail / DMart pricing and margin structures
PRODUCTS = [
    # Breakfast & Bakery (High frequency, varied margins)
    {"id": "P101", "name": "Premium Wheat Bread", "category": "Breakfast & Dairy", "cost": 22.0, "price": 28.0, "high_volume": True},
    {"id": "P102", "name": "Creamy Butter 100g", "category": "Breakfast & Dairy", "cost": 42.0, "price": 52.0, "high_volume": True},
    {"id": "P103", "name": "Mixed Fruit Jam 500g", "category": "Breakfast & Dairy", "cost": 85.0, "price": 140.0, "high_volume": False},
    {"id": "P104", "name": "Crunchy Peanut Butter 350g", "category": "Breakfast & Dairy", "cost": 95.0, "price": 175.0, "high_volume": False},
    {"id": "P105", "name": "Fresh Cow Milk 1L", "category": "Breakfast & Dairy", "cost": 58.0, "price": 64.0, "high_volume": True},
    {"id": "P106", "name": "Corn Flakes Cereal 475g", "category": "Breakfast & Dairy", "cost": 130.0, "price": 210.0, "high_volume": False},
    {"id": "P107", "name": "Premium Instant Coffee 100g", "category": "Breakfast & Dairy", "cost": 140.0, "price": 260.0, "high_volume": False},
    {"id": "P108", "name": "Assam CTC Tea 500g", "category": "Breakfast & Dairy", "cost": 160.0, "price": 230.0, "high_volume": True},

    # Grocery & Staples (High sales, tight margins 8-15%)
    {"id": "P201", "name": "Chakki Fresh Atta 5kg", "category": "Grocery & Staples", "cost": 215.0, "price": 245.0, "high_volume": True},
    {"id": "P202", "name": "Refined Sunflower Oil 1L", "category": "Grocery & Staples", "cost": 128.0, "price": 142.0, "high_volume": True},
    {"id": "P203", "name": "Premium Basmati Rice 5kg", "category": "Grocery & Staples", "cost": 380.0, "price": 490.0, "high_volume": True},
    {"id": "P204", "name": "Toor Dal 1kg", "category": "Grocery & Staples", "cost": 145.0, "price": 165.0, "high_volume": True},
    {"id": "P205", "name": "Refined Pure Sugar 1kg", "category": "Grocery & Staples", "cost": 42.0, "price": 48.0, "high_volume": True},
    {"id": "P206", "name": "Tata Iodized Salt 1kg", "category": "Grocery & Staples", "cost": 22.0, "price": 26.0, "high_volume": True},

    # Snacks & Beverages (Medium sales, higher margins 25-45%)
    {"id": "P301", "name": "Chocolate Chip Cookies 200g", "category": "Snacks & Munchies", "cost": 32.0, "price": 60.0, "high_volume": True},
    {"id": "P302", "name": "Crispy Potato Wafers 150g", "category": "Snacks & Munchies", "cost": 20.0, "price": 40.0, "high_volume": True},
    {"id": "P303", "name": "Sparkling Cola Can 300ml", "category": "Snacks & Munchies", "cost": 22.0, "price": 40.0, "high_volume": True},
    {"id": "P304", "name": "Mixed Roasted Dry Fruits 250g", "category": "Snacks & Munchies", "cost": 220.0, "price": 399.0, "high_volume": False},

    # Personal Care & Cleaning (Attractive profit margins 30-55%)
    {"id": "P401", "name": "Anti-Dandruff Shampoo 400ml", "category": "Personal Care", "cost": 180.0, "price": 320.0, "high_volume": True},
    {"id": "P402", "name": "Nourishing Conditioner 200ml", "category": "Personal Care", "cost": 110.0, "price": 240.0, "high_volume": False},
    {"id": "P403", "name": "Moisturizing Bath Soap 4-Pack", "category": "Personal Care", "cost": 110.0, "price": 165.0, "high_volume": True},
    {"id": "P404", "name": "Herbal Toothpaste 200g", "category": "Personal Care", "cost": 70.0, "price": 120.0, "high_volume": True},
    {"id": "P405", "name": "Liquid Laundry Detergent 1L", "category": "Household Care", "cost": 130.0, "price": 215.0, "high_volume": True},
    {"id": "P406", "name": "Fabric Softener Conditioner 800ml", "category": "Household Care", "cost": 110.0, "price": 225.0, "high_volume": False},
    {"id": "P407", "name": "Citrus Dishwash Gel 750ml", "category": "Household Care", "cost": 85.0, "price": 145.0, "high_volume": True},

    # Electronics & Gadgets (High ticket, high cross-sell utility)
    {"id": "P501", "name": "Intel Core i5 Thin Laptop", "category": "Electronics", "cost": 42000.0, "price": 49990.0, "high_volume": False},
    {"id": "P502", "name": "Wireless Ergonomic Mouse", "category": "Electronics", "cost": 380.0, "price": 899.0, "high_volume": True},
    {"id": "P503", "name": "Padded Laptop Backpack 15.6 inch", "category": "Electronics", "cost": 450.0, "price": 1299.0, "high_volume": True},
    {"id": "P504", "name": "Backlit Mechanical Keyboard", "category": "Electronics", "cost": 950.0, "price": 2199.0, "high_volume": False},
    {"id": "P505", "name": "5G Smartphone 128GB", "category": "Electronics", "cost": 15500.0, "price": 18499.0, "high_volume": False},
    {"id": "P506", "name": "Fast 65W GaN Charger", "category": "Electronics", "cost": 420.0, "price": 1199.0, "high_volume": True},
    {"id": "P507", "name": "20000mAh Dual Output Power Bank", "category": "Electronics", "cost": 750.0, "price": 1699.0, "high_volume": True},
    {"id": "P508", "name": "Active Noise Cancelling Earbuds", "category": "Electronics", "cost": 990.0, "price": 2499.0, "high_volume": False},
    {"id": "P509", "name": "Fitness Smartwatch with SpO2", "category": "Electronics", "cost": 1150.0, "price": 2799.0, "high_volume": False}
]

# Basket co-occurrence profiles representing realistic buyer personas
CO_OCCURRENCE_PROFILES = [
    # Laptop Bundle: Laptop -> Mouse, Laptop Bag, Keyboard
    {
        "primary": "Intel Core i5 Thin Laptop",
        "companions": ["Wireless Ergonomic Mouse", "Padded Laptop Backpack 15.6 inch", "Backlit Mechanical Keyboard"],
        "affinity": 0.88
    },
    # Smartphone Bundle: Phone -> Charger, Power Bank, Earbuds
    {
        "primary": "5G Smartphone 128GB",
        "companions": ["Fast 65W GaN Charger", "20000mAh Dual Output Power Bank", "Active Noise Cancelling Earbuds"],
        "affinity": 0.85
    },
    # Breakfast Combo: Bread -> Butter, Jam, Peanut Butter, Milk
    {
        "primary": "Premium Wheat Bread",
        "companions": ["Creamy Butter 100g", "Mixed Fruit Jam 500g", "Crunchy Peanut Butter 350g", "Fresh Cow Milk 1L"],
        "affinity": 0.82
    },
    # Hair Care Bundle: Shampoo -> Conditioner
    {
        "primary": "Anti-Dandruff Shampoo 400ml",
        "companions": ["Nourishing Conditioner 200ml", "Moisturizing Bath Soap 4-Pack"],
        "affinity": 0.78
    },
    # Laundry Bundle: Detergent -> Fabric Softener, Dishwash
    {
        "primary": "Liquid Laundry Detergent 1L",
        "companions": ["Fabric Softener Conditioner 800ml", "Citrus Dishwash Gel 750ml"],
        "affinity": 0.75
    },
    # Daily Staples Bundle: Atta -> Oil, Rice, Dal, Salt
    {
        "primary": "Chakki Fresh Atta 5kg",
        "companions": ["Refined Sunflower Oil 1L", "Toor Dal 1kg", "Tata Iodized Salt 1kg", "Premium Basmati Rice 5kg"],
        "affinity": 0.80
    },
    # Evening Snack Bundle: Tea -> Sugar, Cookies, Biscuits
    {
        "primary": "Assam CTC Tea 500g",
        "companions": ["Refined Pure Sugar 1kg", "Chocolate Chip Cookies 200g", "Fresh Cow Milk 1L"],
        "affinity": 0.83
    },
    # Wearable Gadgets Bundle: Smartwatch -> Earbuds
    {
        "primary": "Fitness Smartwatch with SpO2",
        "companions": ["Active Noise Cancelling Earbuds", "20000mAh Dual Output Power Bank"],
        "affinity": 0.72
    }
]

def generate_dmart_transactions(num_invoices=2500, output_path="data/raw_dmart_transactions.csv"):
    """
    Generates a realistic transaction log for Market Basket Analysis.
    """
    random.seed(42)
    np.random.seed(42)
    
    prod_map = {p["name"]: p for p in PRODUCTS}
    all_names = list(prod_map.keys())
    
    start_date = datetime(2026, 1, 1)
    records = []
    
    for inv_id in range(1001, 1001 + num_invoices):
        invoice_no = f"INV{inv_id}"
        txn_date = start_date + timedelta(days=random.randint(0, 180), minutes=random.randint(0, 1400))
        cust_age = int(np.clip(np.random.normal(34, 12), 18, 70))
        
        basket_items = set()
        
        # Decide if this transaction triggers a co-occurrence pattern
        applied_pattern = False
        for profile in CO_OCCURRENCE_PROFILES:
            if random.random() < 0.22:  # chance to initiate this profile
                basket_items.add(profile["primary"])
                for comp in profile["companions"]:
                    if random.random() < profile["affinity"]:
                        basket_items.add(comp)
                applied_pattern = True
                break
        
        # Add random items to mimic natural cart noise (basket size 2 to 7)
        target_size = random.randint(2, 6)
        while len(basket_items) < target_size:
            # Weighted choice: high volume items are picked more often
            picked = random.choice(all_names)
            p_obj = prod_map[picked]
            if p_obj.get("high_volume") or random.random() < 0.35:
                basket_items.add(picked)
        
        # Create row entries for each item in the basket
        for item_name in basket_items:
            p_data = prod_map[item_name]
            qty = 1 if p_data["category"] == "Electronics" else random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]
            
            # Inject a few intentional slight dirty inputs to demonstrate Step 2 Data Cleaning
            cost = p_data["cost"]
            price = p_data["price"]
            
            records.append({
                "InvoiceNo": invoice_no,
                "Date": txn_date.strftime("%Y-%m-%d %H:%M"),
                "CustomerID": f"CUST_{random.randint(100, 800)}",
                "CustomerAge": cust_age,
                "ProductID": p_data["id"],
                "ProductName": item_name,
                "Category": p_data["category"],
                "Quantity": qty,
                "CostPrice": cost,
                "SellingPrice": price
            })
            
    df = pd.DataFrame(records)
    
    # Intentionally add a tiny fraction of duplicates & nulls so cleaning step works authentically
    sample_dups = df.sample(n=12, random_state=42)
    df = pd.concat([df, sample_dups], ignore_index=True)
    
    # Introduce 5 null values in CustomerAge
    df.loc[df.sample(n=5, random_state=42).index, "CustomerAge"] = np.nan
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} transaction records across {num_invoices} invoices.")
    print(f"📁 Saved to: {output_path}")
    return df

if __name__ == "__main__":
    generate_dmart_transactions()
