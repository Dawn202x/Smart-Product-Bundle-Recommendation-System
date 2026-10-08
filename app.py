"""
Smart Product Bundle Recommendation System
Streamlit Web Application with Visual Product Cards & Role-Based Access Control:
- Visual Customer Portal: Product Images, Interactive Bundle Combiner, Visual Catalog, Cart Simulator
- Admin Portal: Confidential Business Analytics, Volume vs Margin Matrix, Association Rules Explorer, Margin Weight Tuning
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("."))
import time
from src.profit_aware_recommender import ProfitAwareRecommender
from src.apriori_engine import AprioriEngine

st.set_page_config(
    page_title="SmartCart AI - Visual Bundle Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Product Image Mapping (High quality retail imagery)
PRODUCT_IMAGES = {
    # Electronics
    "Intel Core i5 Thin Laptop": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=500",
    "Wireless Ergonomic Mouse": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=500",
    "Padded Laptop Backpack 15.6 inch": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=500",
    "Backlit Mechanical Keyboard": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=500",
    "5G Smartphone 128GB": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=500",
    "Fast 65W GaN Charger": "https://images.unsplash.com/photo-1583863788434-e58a36330cf0?w=500",
    "20000mAh Dual Output Power Bank": "https://images.unsplash.com/photo-1609592424817-2b3d3c8b8e0f?w=500",
    "Active Noise Cancelling Earbuds": "https://images.unsplash.com/photo-1606220945770-b5b6c2c55bf1?w=500",
    "Fitness Smartwatch with SpO2": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500",
    
    # Breakfast & Dairy
    "Premium Wheat Bread": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=500",
    "Creamy Butter 100g": "https://images.unsplash.com/photo-1589985270826-4b7bb135bc9d?w=500",
    "Crunchy Peanut Butter 350g": "https://images.unsplash.com/photo-1589367920969-ab8e050bbb04?w=500",
    "Mixed Fruit Jam 500g": "https://images.unsplash.com/photo-1590412200988-a436970781fa?w=500",
    "Fresh Cow Milk 1L": "https://images.unsplash.com/photo-1550583724-b2692b85b150?w=500",
    "Corn Flakes Cereal 475g": "https://images.unsplash.com/photo-1584473457406-6240486418e9?w=500",
    "Premium Instant Coffee 100g": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=500",
    "Assam CTC Tea 500g": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?w=500",

    # Staples & Grocery
    "Chakki Fresh Atta 5kg": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=500",
    "Refined Sunflower Oil 1L": "https://images.unsplash.com/photo-1474979266404-7eaacbcd87c5?w=500",
    "Premium Basmati Rice 5kg": "https://images.unsplash.com/photo-1586201375761-83865001e31c?w=500",
    "Toor Dal 1kg": "https://images.unsplash.com/photo-1515543237350-b3eea1ec8082?w=500",
    "Refined Pure Sugar 1kg": "https://images.unsplash.com/photo-1622484216850-d463e26aa75e?w=500",
    "Tata Iodized Salt 1kg": "https://images.unsplash.com/photo-1518110925495-5fe2fda0442c?w=500",

    # Snacks
    "Chocolate Chip Cookies 200g": "https://images.unsplash.com/photo-1558961363-fa8fdf82db35?w=500",
    "Crispy Potato Wafers 150g": "https://images.unsplash.com/photo-1566478989037-eec170784d0b?w=500",
    "Sparkling Cola Can 300ml": "https://images.unsplash.com/photo-1629203849820-fdd70d49c38e?w=500",
    "Mixed Roasted Dry Fruits 250g": "https://images.unsplash.com/photo-1608797178974-15b35a64ede9?w=500",

    # Personal Care & Household
    "Anti-Dandruff Shampoo 400ml": "https://images.unsplash.com/photo-1535585209827-a15fcdbc4c2d?w=500",
    "Nourishing Conditioner 200ml": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?w=500",
    "Moisturizing Bath Soap 4-Pack": "https://images.unsplash.com/photo-1607006314592-808620716616?w=500",
    "Herbal Toothpaste 200g": "https://images.unsplash.com/photo-1559599101-f09722fb4948?w=500",
    "Liquid Laundry Detergent 1L": "https://images.unsplash.com/photo-1585421514738-01798e348b17?w=500",
    "Fabric Softener Conditioner 800ml": "https://images.unsplash.com/photo-1517677208171-0bc6725a3e60?w=500",
    "Citrus Dishwash Gel 750ml": "https://images.unsplash.com/photo-1583947215259-38e31be8751f?w=500",
}

def get_product_image(name):
    return PRODUCT_IMAGES.get(name, "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=500")

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #7b1fa2;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .role-badge-admin {
        background: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .role-badge-user {
        background: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .hero-product-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .catalog-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 18px;
    }
    .smart-card {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
    }
    .trad-card {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
    }
    .badge-profit {
        background: #16803c;
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .bundle-math-box {
        background: #faf5ff;
        border: 1px dashed #7b1fa2;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-top: 15px;
    }
    .lock-card {
        background: #fff1f2;
        border: 2px dashed #e11d48;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
    st.session_state.role = "customer"  # default role
    st.session_state.username = "Customer_User"
if "selected_product_global" not in st.session_state:
    st.session_state.selected_product_global = "Intel Core i5 Thin Laptop"

@st.cache_resource
def get_recommender():
    return ProfitAwareRecommender(
        rules_path="data/association_rules.csv",
        products_path="data/cleaned_dmart_dataset.csv"
    )

@st.cache_data
def load_datasets():
    clean_df = pd.read_csv("data/cleaned_dmart_dataset.csv") if os.path.exists("data/cleaned_dmart_dataset.csv") else None
    rules_df = pd.read_csv("data/association_rules.csv") if os.path.exists("data/association_rules.csv") else None
    summary_df = pd.read_csv("data/eda_product_summary.csv") if os.path.exists("data/eda_product_summary.csv") else None
    return clean_df, rules_df, summary_df

recommender = get_recommender()
clean_df, rules_df, summary_df = load_datasets()

# ==============================================================================
# SIDEBAR AUTHENTICATION & NAVIGATION
# ==============================================================================
st.sidebar.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=400", use_container_width=True)
st.sidebar.title("SmartCart AI")

# Role Status Banner
if st.session_state.role == "admin":
    st.sidebar.markdown("**Status:** <span class='role-badge-admin'>👑 ADMIN PORTAL</span>", unsafe_allow_html=True)
    st.sidebar.caption(f"Active Account: **{st.session_state.username}** (Full Access)")
else:
    st.sidebar.markdown("**Status:** <span class='role-badge-user'>👤 CUSTOMER PORTAL</span>", unsafe_allow_html=True)
    st.sidebar.caption(f"Active Account: **{st.session_state.username}** (Shopper View)")

st.sidebar.markdown("---")

# Quick Role Switcher
with st.sidebar.expander("🔐 Switch User / Admin Login", expanded=False):
    tab_auth_user, tab_auth_admin = st.tabs(["👤 Customer", "👑 Store Admin"])
    
    with tab_auth_user:
        u_name = st.text_input("Customer Name:", value="Ramesh Kumar", key="cust_name_input")
        if st.button("Login as Customer", key="btn_user_login", use_container_width=True):
            st.session_state.role = "customer"
            st.session_state.username = u_name if u_name else "Customer_User"
            st.rerun()
            
    with tab_auth_admin:
        admin_user = st.text_input("Admin Username:", value="admin", key="admin_user_input")
        admin_pass = st.text_input("Admin Password:", value="admin123", type="password", key="admin_pass_input")
        if st.button("Login as Admin", key="btn_admin_login", use_container_width=True):
            if admin_user == "admin" and admin_pass == "admin123":
                st.session_state.role = "admin"
                st.session_state.username = "Store Administrator"
                st.success("Admin authenticated!")
                st.rerun()
            else:
                st.error("Invalid credentials! (admin / admin123)")

st.sidebar.markdown("### Navigation")

# Dynamic Menu based on Role
if st.session_state.role == "admin":
    menu = st.sidebar.radio(
        "Management Console:",
        [
            "🔬 Apriori Association Engine",
            "📊 Business Analytics & EDA",
            "🎯 Product Portfolio Matrix (Volume vs. Margin)",
            "🔍 Association Rules Explorer",
            "⚙️ Bundle Optimization & Weight Tuning",
            "🧹 Data Cleaning & Preprocessing Pipeline",
            "📖 System Architecture & Model Guide"
        ]
    )
else:
    # Customer Menu: clean shopping navigation
    menu = st.sidebar.radio(
        "Shopping Menu:",
        [
            "🎯 Product Recommendations & Bundles",
            "🛍️ Store Catalog",
            "🛒 Shopping Cart"
        ]
    )

st.sidebar.markdown("---")
st.sidebar.caption("SmartCart AI • Retail Recommendation & Profit Optimization Platform")

# ==============================================================================
# VIEW 1: PRODUCT RECOMMENDATIONS & BUNDLES (CUSTOMER VIEW)
# ==============================================================================
if menu == "🎯 Product Recommendations & Bundles":
    st.markdown('<div class="main-header">Product Recommendations & Bundles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalized companion packages with instant bundle savings</div>', unsafe_allow_html=True)
    
    if recommender.products_df is None:
        st.error("Dataset not found. Please ensure data/cleaned_dmart_dataset.csv is present.")
    else:
        products_list = recommender.get_all_products()
        
        # Product selector bar
        col_p, col_k = st.columns([3, 1])
        with col_p:
            default_prod = st.session_state.selected_product_global
            default_idx = products_list.index(default_prod) if default_prod in products_list else 0
            selected_product = st.selectbox("I am shopping for:", products_list, index=default_idx)
            st.session_state.selected_product_global = selected_product
        with col_k:
            top_k = st.slider("Companion items in bundle:", min_value=1, max_value=3, value=3)
            
        result = recommender.recommend(
            selected_product=selected_product,
            top_k=top_k,
            w_lift=0.40,
            w_margin=0.40,
            w_confidence=0.20,
            bundle_discount_pct=5.0
        )
        
        if result:
            base = result["selected_product"]
            smart = result["smart_bundle"]
            
            # 1. VISUAL HERO CARD FOR SELECTED PRIMARY ITEM
            st.markdown('<div class="hero-product-card">', unsafe_allow_html=True)
            col_hero_img, col_hero_details = st.columns([1, 2.8])
            
            with col_hero_img:
                st.image(get_product_image(base["name"]), caption=base["name"], use_container_width=True)
                
            with col_hero_details:
                st.markdown(f"<span style='background:#e9d5ff;color:#6b21a8;font-weight:bold;padding:3px 8px;border-radius:4px;font-size:0.8rem;'>{base['category']}</span>", unsafe_allow_html=True)
                st.markdown(f"## {base['name']}")
                st.markdown(f"⭐ **4.8 / 5.0** (Verified Buyer Rating • Fast Free Delivery Available)")
                st.markdown(f"<h3 style='color:#7b1fa2;margin-top:8px;'>Price: ₹{base['price']:,.2f}</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#555;'>In stock & ready to dispatch from local DMart fulfillment center.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Frequently Bought Together")
            
            flow_cols = st.columns(len(smart["recommended_items"]) * 2 + 1)
            
            # Primary Product Image
            with flow_cols[0]:
                st.image(get_product_image(base["name"]), use_container_width=True)
                st.caption(f"**{base['name'].split()[0]}**<br>₹{base['price']:,.0f}", unsafe_allow_html=True)
                
            col_idx = 1
            for item in smart["recommended_items"]:
                with flow_cols[col_idx]:
                    st.markdown("<h2 style='text-align:center;color:#7b1fa2;margin-top:25px;'>+</h2>", unsafe_allow_html=True)
                col_idx += 1
                with flow_cols[col_idx]:
                    st.image(get_product_image(item["name"]), use_container_width=True)
                    st.caption(f"**{item['name'].split()[0]}**<br>₹{item['price']:,.0f}", unsafe_allow_html=True)
                col_idx += 1
                
            st.markdown("---")
            
            # 3. INDIVIDUAL COMPANION CARDS WITH PHOTOS
            st.markdown("#### Included in this Package:")
            comp_cols = st.columns(len(smart["recommended_items"]))
            for idx, item in enumerate(smart["recommended_items"]):
                with comp_cols[idx]:
                    st.markdown('<div class="catalog-card">', unsafe_allow_html=True)
                    st.image(get_product_image(item["name"]), use_container_width=True)
                    st.markdown(f"**{item['name']}**")
                    st.markdown(f"<span style='color:#16803c;font-weight:bold;font-size:1.1rem;'>₹{item['price']:,.2f}</span>", unsafe_allow_html=True)
                    st.caption("✓ 100% Genuine • In Stock • Free Delivery")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            # 4. BUNDLE SAVINGS & CHECKOUT ACTION
            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                st.markdown('<div class="smart-card">', unsafe_allow_html=True)
                st.markdown("### Bundle Price Summary")
                st.write(f"Total Individual Price: **₹{smart['individual_sum_price']:,.2f}**")
                st.write(f"Package Savings (5% Instant Waiver): <span style='color:#16803c;font-weight:bold;'>-₹{smart['bundle_discount_amount']:,.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"## Bundle Price: <span style='color:#7b1fa2;'>₹{smart['bundle_price']:,.2f}</span>", unsafe_allow_html=True)
                if st.button("🛒 Add All Bundle Items to Cart", type="primary", use_container_width=True):
                    st.balloons()
                    st.success(f"🎉 Successfully added bundle to cart! You saved ₹{smart['bundle_discount_amount']:,.2f}!")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with b_col2:
                st.markdown("""
                <div class="hero-product-card" style="border-left: 5px solid #16803c;">
                    <h4>Why Buy as a Bundle?</h4>
                    <ul style="font-size: 0.95rem; color: #4b5563; line-height: 1.8;">
                        <li><b>Instant 5% Package Savings</b></li>
                        <li><b>Verified Complementary Products</b></li>
                        <li><b>Free Priority Delivery</b> for complete package</li>
                        <li>DMart 7-Day Hassle-Free Returns</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: BROWSE PRODUCTS WITH VISUAL PRODUCT GRID
# ==============================================================================
elif menu == "🛍️ Store Catalog":
    st.markdown('<div class="main-header">🛍️ Store Catalog</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse all products and generate instant recommendations</div>', unsafe_allow_html=True)
    
    if clean_df is not None:
        cat_filter = st.selectbox("Filter by Department:", ["All Departments"] + sorted(clean_df["Category"].unique().tolist()))
        
        display_df = clean_df.copy()
        if cat_filter != "All Departments":
            display_df = display_df[display_df["Category"] == cat_filter]
            
        unique_prods = display_df.groupby("ProductName").agg(
            Category=("Category", "first"),
            Price=("SellingPrice", "first")
        ).reset_index()
        
        # Render responsive 4-column visual product grid
        cols_per_row = 4
        num_items = len(unique_prods)
        
        for r in range(0, num_items, cols_per_row):
            row_items = unique_prods.iloc[r:r+cols_per_row]
            grid_cols = st.columns(cols_per_row)
            for c, (_, item) in enumerate(row_items.iterrows()):
                with grid_cols[c]:
                    st.markdown('<div class="catalog-card">', unsafe_allow_html=True)
                    st.image(get_product_image(item["ProductName"]), use_container_width=True)
                    st.markdown(f"**{item['ProductName']}**")
                    st.caption(item["Category"])
                    st.markdown(f"<span style='color:#7b1fa2;font-size:1.15rem;font-weight:bold;'>₹{item['Price']:,.2f}</span>", unsafe_allow_html=True)
                    
                    if st.button("⚡ View Recommendations", key=f"btn_cat_{r}_{c}", use_container_width=True):
                        st.session_state.selected_product_global = item["ProductName"]
                        st.success(f"Selected: {item['ProductName']}! Switch to '🎯 Product Recommendations & Bundles' to view.")
                    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 3: MY SHOPPING CART WITH PRODUCT IMAGES
# ==============================================================================
elif menu == "🛒 Shopping Cart":
    st.markdown('<div class="main-header">🛒 Shopping Cart</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review your selected items and package savings</div>', unsafe_allow_html=True)
    
    col_cart_items, col_cart_summary = st.columns([1.8, 1.2])
    
    cart_items = [
        {"name": "Intel Core i5 Thin Laptop", "price": 49990.0},
        {"name": "Wireless Ergonomic Mouse", "price": 899.0},
        {"name": "Padded Laptop Backpack 15.6 inch", "price": 1299.0},
        {"name": "Backlit Mechanical Keyboard", "price": 2199.0}
    ]
    
    with col_cart_items:
        st.markdown("### Items in Cart (4 Products)")
        for item in cart_items:
            c_img, c_desc, c_pr = st.columns([1, 2.5, 1])
            with c_img:
                st.image(get_product_image(item["name"]), width=75)
            with c_desc:
                st.markdown(f"**{item['name']}**")
                st.caption("Qty: 1 • Status: In Stock")
            with c_pr:
                st.markdown(f"**₹{item['price']:,.2f}**")
            st.markdown("---")
            
    with col_cart_summary:
        st.markdown('<div class="smart-card">', unsafe_allow_html=True)
        st.markdown("### Order Summary")
        subtotal = sum(x["price"] for x in cart_items)
        discount = subtotal * 0.05
        total = subtotal - discount
        
        st.write(f"Subtotal: **₹{subtotal:,.2f}**")
        st.write(f"Delivery: <span style='color:green;font-weight:bold;'>FREE</span>", unsafe_allow_html=True)
        st.write(f"Smart Bundle Savings (5%): <span style='color:green;font-weight:bold;'>-₹{discount:,.2f}</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"## Total Due: <span style='color:#7b1fa2;'>₹{total:,.2f}</span>", unsafe_allow_html=True)
        
        if st.button("💳 Proceed to Payment", type="primary", use_container_width=True):
            st.balloons()
            st.success("🎉 Payment Confirmed! Your order will be delivered tomorrow.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW: ADMIN - APRIORI ASSOCIATION ENGINE & MINING
# ==============================================================================
elif menu == "🔬 Apriori Association Engine":
    st.markdown('<div class="main-header">🔬 Apriori Association Engine & Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Market Basket Analysis: Mine Frequent Itemsets and Association Rules from Cleaned Transactions</div>', unsafe_allow_html=True)
    
    col_ap1, col_ap2, col_ap3 = st.columns(3)
    with col_ap1:
        ui_min_sup = st.slider("Minimum Support (min_support):", 0.005, 0.050, 0.012, 0.001, 
                               help="Frequency threshold: Fraction of transactions that must contain the itemset.")
    with col_ap2:
        ui_min_conf = st.slider("Minimum Confidence (min_confidence):", 0.10, 0.90, 0.20, 0.05,
                                help="Reliability threshold: Probability of buying B when A is in basket.")
    with col_ap3:
        ui_min_lift = st.slider("Minimum Lift (min_lift):", 1.0, 5.0, 1.1, 0.1,
                                help="Affinity threshold: Lift > 1.0 indicates true complementary association.")
                                
    if st.button("▶️ Execute Apriori Algorithm Live on Cleaned Transactions", type="primary", use_container_width=True):
        if clean_df is not None:
            with st.spinner("Running Apriori candidate generation, pruning, and association rule mining..."):
                t_start = time.time()
                live_engine = AprioriEngine(min_support=ui_min_sup, min_confidence=ui_min_conf, min_lift=ui_min_lift)
                raw_rules = live_engine.fit(clean_df)
                enriched_rules = live_engine.enrich_with_economics(raw_rules, clean_df)
                t_elapsed = (time.time() - t_start) * 1000  # in ms
                
                # Save live rules
                enriched_rules.to_csv("data/association_rules.csv", index=False)
                st.session_state["live_apriori_run"] = {
                    "time_ms": t_elapsed,
                    "stats": live_engine.stats,
                    "frequent_df": live_engine.get_frequent_itemsets_df(),
                    "rules_df": enriched_rules
                }
                st.success(f"🎉 Apriori Algorithm completed in {t_elapsed:.1f} ms! Mined {len(enriched_rules)} rules.")
                
    # Display results if already executed or load default
    if "live_apriori_run" in st.session_state:
        run_data = st.session_state["live_apriori_run"]
        stats = run_data["stats"]
        
        st.markdown("---")
        st.markdown("### 📊 Apriori Mining Execution Statistics:")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Multi-item Baskets", f"{stats['total_baskets']:,}")
        k2.metric("Frequent 1-Items (L1)", f"{stats['l1_frequent']} / {stats['c1_candidates']}")
        k3.metric("Frequent 2-Pairs (L2)", f"{stats['l2_frequent']} / {stats['c2_candidates']}")
        k4.metric("Frequent 3-Trios (L3)", f"{stats['l3_frequent']} / {stats['c3_candidates']}")
        k5.metric("Valid Rules Mined", f"{len(run_data['rules_df']):,}", delta=f"{run_data['time_ms']:.0f} ms")
        
        tab_itemsets, tab_rules, tab_math = st.tabs([
            "📦 Frequent Itemsets (L1, L2, L3)",
            "📜 Mined Association Rules Table",
            "📐 Algorithmic Formulation & Principles"
        ])
        
        with tab_itemsets:
            st.markdown("#### Frequent Itemsets Generated by Level ($L_k$):")
            st.caption("Each itemset passed the Minimum Support threshold across all multi-item transactions.")
            st.dataframe(run_data["frequent_df"], use_container_width=True)
            
        with tab_rules:
            st.markdown(f"#### Association Rules Meeting Thresholds ({len(run_data['rules_df'])} Rules):")
            st.caption("Ranked by Lift (Complementary Association Strength)")
            disp_cols = ["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Consequent_Category", "Consequent_Price", "Consequent_Margin_Pct", "Bundle_Profit"]
            st.dataframe(run_data["rules_df"][disp_cols].sort_values(by="Lift", ascending=False), use_container_width=True)
            
        with tab_math:
            st.markdown("""
            ### 📐 Apriori Algorithm Mathematical Foundation:
            
            1. **The Apriori Pruning Property:**
               $$\\text{If an itemset } I \\text{ is frequent, all of its subsets must also be frequent.}$$
               $$\\text{Contrapositive: If a subset } S \\subset I \\text{ is infrequent, } I \\text{ can never be frequent (pruned without counting).}$$
               
            2. **Candidate Generation & Support Counting:**
               $$\\text{Support}(A \\to B) = P(A \\cap B) = \\frac{\\text{Transactions containing both } A \\text{ and } B}{N_{\\text{total}}}$$
               
            3. **Confidence Calculation:**
               $$\\text{Confidence}(A \\to B) = P(B \\mid A) = \\frac{\\text{Support}(A \\cup B)}{\\text{Support}(A)}$$
               
            4. **Lift Calculation (Association Strength):**
               $$\\text{Lift}(A \\to B) = \\frac{\\text{Confidence}(A \\to B)}{\\text{Support}(B)} = \\frac{P(A \\cap B)}{P(A) \\times P(B)}$$
               - $\\text{Lift} = 1$: Items occur together purely by random chance.
               - $\\text{Lift} > 1$: Strong positive complementary affinity (Basis for our bundles!).
            """)
    else:
        st.info("👆 Click the button above to execute the Apriori algorithm live and inspect intermediate frequent itemsets ($L_1, L_2, L_3$) and association rules.")

# ==============================================================================
# VIEW: ADMIN - BUSINESS ANALYTICS & EDA
# ==============================================================================
elif menu == "📊 Business Analytics & EDA":
    st.markdown('<div class="main-header">📊 Retail Business Analytics & EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Store Financial Metrics and Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if clean_df is not None:
        total_rev = clean_df["Revenue"].sum()
        total_profit = clean_df["Profit"].sum()
        total_txns = clean_df["InvoiceNo"].nunique()
        aov = total_rev / total_txns
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Store Revenue", f"₹{total_rev/100000:,.1f} Lakhs")
        m2.metric("Total Gross Profit", f"₹{total_profit/100000:,.1f} Lakhs", delta=f"{(total_profit/total_rev)*100:.1f}% Margin")
        m3.metric("Total Transactions (Invoices)", f"{total_txns:,}")
        m4.metric("Average Order Value (AOV)", f"₹{aov:,.2f}")
        
        st.markdown("---")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("🏆 Top 10 Products by Sales Volume")
            if os.path.exists("data/eda_charts/top_selling_products.png"):
                st.image("data/eda_charts/top_selling_products.png", use_container_width=True)
            else:
                top_v = clean_df.groupby("ProductName")["Quantity"].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_v)
                
        with col_c2:
            st.subheader("💰 Top 10 Products by Total Profit (₹)")
            if os.path.exists("data/eda_charts/top_profitable_products.png"):
                st.image("data/eda_charts/top_profitable_products.png", use_container_width=True)
            else:
                top_p = clean_df.groupby("ProductName")["Profit"].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_p)
                
        st.markdown("---")
        st.subheader("📦 Category Financial Breakdown (Revenue, Profit, Margins)")
        cat_df = clean_df.groupby("Category").agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units_Sold=("Quantity", "sum")
        ).reset_index()
        cat_df["Profit_Margin_%"] = ((cat_df["Profit"] / cat_df["Revenue"]) * 100).round(2)
        cat_df["Revenue_Share_%"] = ((cat_df["Revenue"] / total_rev) * 100).round(2)
        st.dataframe(cat_df.sort_values(by="Revenue", ascending=False), use_container_width=True)

# ==============================================================================
# VIEW: ADMIN - PRODUCT PORTFOLIO MATRIX
# ==============================================================================
elif menu == "🎯 Product Portfolio Matrix (Volume vs. Margin)":
    st.markdown('<div class="main-header">🎯 Product Portfolio Matrix (Volume vs. Margin)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strategic Portfolio Segmentation: Categorizing items into Traffic Drivers vs High-Margin Companions</div>', unsafe_allow_html=True)
    
    if os.path.exists("data/eda_charts/sales_vs_profit_margin_matrix.png"):
        st.image("data/eda_charts/sales_vs_profit_margin_matrix.png", use_container_width=True)
        
    st.markdown("""
    ### 🧠 Retail Strategic Interpretation:
    1. **High Volume + Low Margin (Traffic Drivers / Staples):**
       - Examples: *Fresh Cow Milk (9.4%), Refined Sunflower Oil (9.9%), Chakki Atta (12.2%), Bread (21.4%)*
       - **Strategy:** Essential everyday goods that drive footfall and transaction volume. Never bundle only low-margin staples together.
    2. **High Volume + High Margin (Star Bundles):**
       - Examples: *Wireless Ergonomic Mouse (57.7%), Fast GaN Chargers (65.0%), Padded Laptop Backpacks (65.4%)*
       - **Strategy:** Top-performing products that generate strong revenue and healthy margins. Ideal bundle anchors.
    3. **Low Volume + High Margin (High-Margin Companions):**
       - Examples: *Crunchy Peanut Butter (45.7%), Mixed Fruit Jam (39.3%), Fabric Softener (51.1%)*
       - **Strategy:** High-margin complementary items to cross-sell alongside volume drivers to elevate total basket profit.
    """)

# ==============================================================================
# VIEW: ADMIN - ASSOCIATION RULES EXPLORER
# ==============================================================================
elif menu == "🔍 Association Rules Explorer":
    st.markdown('<div class="main-header">🔍 Association Rules Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Filter and analyze mined association rules across Support, Confidence, Lift, and Margin %</div>', unsafe_allow_html=True)
    
    if rules_df is not None:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_lift_val = st.slider("Minimum Lift:", float(rules_df["Lift"].min()), float(rules_df["Lift"].max()), 1.5, 0.1)
        with col_f2:
            min_conf_val = st.slider("Minimum Confidence:", float(rules_df["Confidence"].min()), 1.0, 0.30, 0.05)
        with col_f3:
            min_margin_val = st.slider("Minimum Consequent Margin %:", float(rules_df["Consequent_Margin_Pct"].min()), float(rules_df["Consequent_Margin_Pct"].max()), 20.0, 5.0)
            
        filtered_rules = rules_df[
            (rules_df["Lift"] >= min_lift_val) &
            (rules_df["Confidence"] >= min_conf_val) &
            (rules_df["Consequent_Margin_Pct"] >= min_margin_val)
        ].copy()
        
        st.write(f"Showing **{len(filtered_rules)}** association rules matching criteria:")
        display_cols = ["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Consequent_Category", "Consequent_Price", "Consequent_Profit", "Consequent_Margin_Pct", "Bundle_Profit"]
        st.dataframe(filtered_rules[display_cols].sort_values(by="Lift", ascending=False), use_container_width=True)

# ==============================================================================
# VIEW: ADMIN - BUNDLE OPTIMIZATION & WEIGHT TUNING
# ==============================================================================
elif menu == "⚙️ Bundle Optimization & Weight Tuning":
    st.markdown('<div class="main-header">⚙️ Bundle Optimization & Weight Tuning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure multi-objective ranking parameters (Lift vs Profit Margin) and compare bundle outputs</div>', unsafe_allow_html=True)
    
    products_list = recommender.get_all_products()
    col_prod, col_k = st.columns([3, 1])
    with col_prod:
        selected_product = st.selectbox("Select Target Product to Simulate:", products_list, index=0)
    with col_k:
        top_k = st.slider("Companion Items Count:", 1, 4, 3)
        
    with st.expander("⚙️ Algorithm Weight Tuning (Multi-Objective Function)", expanded=True):
        col_w1, col_w2, col_w3, col_disc = st.columns(4)
        with col_w1:
            w_lift = st.slider("Lift Weight (α):", 0.0, 1.0, 0.40, 0.05, help="Co-purchase affinity weight")
        with col_w2:
            w_margin = st.slider("Profit Margin Weight (β):", 0.0, 1.0, 0.40, 0.05, help="Margin optimization weight")
        with col_w3:
            w_conf = st.slider("Confidence Weight (γ):", 0.0, 1.0, 0.20, 0.05, help="Occurrence confidence weight")
        with col_disc:
            bundle_disc = st.slider("Bundle Discount (%):", 0.0, 20.0, 5.0, 1.0)
            
    result = recommender.recommend(
        selected_product=selected_product,
        top_k=top_k,
        w_lift=w_lift,
        w_margin=w_margin,
        w_confidence=w_conf,
        bundle_discount_pct=bundle_disc
    )
    
    if result:
        base = result["selected_product"]
        smart = result["smart_bundle"]
        trad = result["traditional_bundle"]
        diff = result["profit_difference"]
        growth = result["profit_growth_pct"]
        
        st.markdown(f"""
        <div class="hero-product-card">
            <span style="font-size: 0.85rem; color: #888;">TARGET ITEM FINANCIALS</span><br>
            <span style="font-size: 1.3rem; font-weight: bold;">{base['name']}</span> ({base['category']}) &nbsp;|&nbsp;
            Price: <b>₹{base['price']:,.2f}</b> &nbsp;|&nbsp; Cost: <b>₹{base['cost']:,.2f}</b> &nbsp;|&nbsp; 
            Unit Profit: <b style="color: #16803c;">₹{base['unit_profit']:,.2f}</b> ({base['margin_pct']:.1f}% Margin)
        </div>
        """, unsafe_allow_html=True)
        
        st.info(result["explanation"])
        
        col_trad, col_smart = st.columns(2)
        with col_trad:
            st.markdown('<div class="trad-card">', unsafe_allow_html=True)
            st.markdown("### 🏷️ Standard Apriori (Frequency Ranking)")
            st.caption("Standard Market Basket Analysis: Ranked purely by Co-purchase Count")
            st.markdown("---")
            for item in trad["recommended_items"]:
                st.markdown(f"- **{item['name']}**<br><small>Price: ₹{item['price']} | Profit: ₹{item['profit']} ({item['margin_pct']:.1f}%) | Lift: {item['lift']:.2f}</small>", unsafe_allow_html=True)
            st.markdown("---")
            st.write(f"Bundle Price: **₹{trad['bundle_price']:,.2f}**")
            st.markdown(f"#### Net Profit: **₹{trad['bundle_profit']:,.2f}** ({trad['bundle_margin_pct']:.1f}% Margin)")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_smart:
            st.markdown('<div class="smart-card">', unsafe_allow_html=True)
            st.markdown("### 🚀 Profit-Optimized Recommender")
            st.caption("Dual-Objective: Balances Co-purchase Lift + High Profit Margin")
            st.markdown("---")
            for item in smart["recommended_items"]:
                st.markdown(f"- **{item['name']}** <span class='badge-profit'>{item['margin_pct']:.1f}% Margin</span><br><small>Price: ₹{item['price']} | Profit: ₹{item['profit']} | Hybrid Score: <b>{item['score']:.3f}</b></small>", unsafe_allow_html=True)
            st.markdown("---")
            st.write(f"Bundle Price: **₹{smart['bundle_price']:,.2f}**")
            st.markdown(f"#### Net Profit: <span style='color:#16803c;'>**₹{smart['bundle_profit']:,.2f}**</span> ({smart['bundle_margin_pct']:.1f}% Margin)", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW: ADMIN - DATA CLEANING PIPELINE INSPECTOR
# ==============================================================================
elif menu == "🧹 Data Cleaning & Preprocessing Pipeline":
    st.markdown('<div class="main-header">🧹 Data Cleaning & Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Internal Dataset Transformations and Hygiene Verification</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 Data Preprocessing Steps Executed:
    1. **Missing Data Scrubbing:** Cleaned missing values in transactions, customer age median imputation.
    2. **Deduplication:** Purged duplicate records to avoid artificial frequency inflation.
    3. **Financial Formulas Applied:**
       $$\\text{Revenue} = \\text{Quantity} \\times \\text{Selling Price}$$
       $$\\text{Total Cost} = \\text{Quantity} \\times \\text{Cost Price}$$
       $$\\text{Profit} = \\text{Revenue} - \\text{Total Cost}$$
       $$\\text{Profit Margin \\%} = \\left(\\frac{\\text{Selling Price} - \\text{Cost Price}}{\\text{Selling Price}}\\right) \\times 100$$
    """)
    if clean_df is not None:
        st.dataframe(clean_df.head(15), use_container_width=True)

# ==============================================================================
# VIEW: ADMIN - SYSTEM ARCHITECTURE & MODEL GUIDE
# ==============================================================================
elif menu == "📖 System Architecture & Model Guide":
    st.markdown('<div class="main-header">📖 System Architecture & Model Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Technical documentation on Market Basket Mining, Multi-Objective Optimization, and Retail Impact</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📌 Core Architecture Overview:
    > *"Traditional Apriori only recommends products customers buy together, ignoring profit margins. Supermarket staples (like bread/milk) carry thin margins (8–15%), while complementary items offer high margins (35–65%). This system implements a **Profit-Optimized Dual-Objective Recommendation Engine** that balances co-purchase affinity (Lift & Confidence) with contribution profit margins, maximizing retailer gross profit while keeping recommendations highly complementary and relevant."*
    """)
    
    st.markdown("---")
    with st.expander("📌 Why Separate Admin and Customer Portals?"):
        st.write("""
        In modern commercial retail systems, unit acquisition costs, supplier wholesale prices, and exact profit margins are strictly confidential business intelligence. 
        Customers only see retail prices, bundle discounts, and savings. Store administrators require access to profitability metrics, margin matrices, and algorithm parameter tuning.
        """)
    with st.expander("📐 Mathematical Formulation of Multi-Objective Scoring"):
        st.write("""
        $$\\text{Score} = \\alpha \\times \\text{Norm}(\\text{Lift}) + \\beta \\times \\text{Norm}(\\text{Margin \\%}) + \\gamma \\times \\text{Confidence}$$
        
        Where:
        - $\\alpha$ (Default 0.40): Weight assigned to complementary co-purchase strength (Lift)
        - $\\beta$ (Default 0.40): Weight assigned to product gross margin percentage
        - $\\gamma$ (Default 0.20): Weight assigned to conditional occurrence probability (Confidence)
        - Normalization uses Min-Max scaling across all candidate rules for the antecedent item.
        """)
    with st.expander("⚡ Apriori Pruning & Frequent Itemset Mining"):
        st.write("""
        The engine employs the downward-closure property (Apriori principle):
        - Any subset of a frequent itemset must be frequent.
        - If an itemset is infrequent, all its supersets will also be infrequent and are pruned immediately.
        - The algorithm mines $L_1$ frequent items, $L_2$ frequent itemsets (pairs), and $L_3$ frequent trios before generating association rules satisfying Minimum Support, Minimum Confidence, and Minimum Lift thresholds.
        """)
    with st.expander("💼 Business & Economic Impact"):
        st.write("""
        - **Higher Average Order Value (AOV):** Packaging high-affinity companion products into a 1-click bundle increases the basket size.
        - **Gross Profit Expansion:** Prioritizing companion items with 40–65% margins over zero-margin staples yields up to **15–30% higher gross profit per transaction**.
        - **Inventory Optimization:** Helps move high-margin inventory by pairing it naturally with high-velocity anchor items.
        """)

