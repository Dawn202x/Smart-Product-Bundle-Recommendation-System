"""
Step 5: Smart Profit-Aware Product Bundle Recommendation Engine
Multi-Objective Optimization:
- Balances co-purchase affinity (Lift/Confidence) with Retailer Profit Margin
- Compares Standard Frequency-based Apriori vs. Profit-Optimized Smart Bundling
- Quantifies bundle savings for the customer and profit gains for the retailer
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

class ProfitAwareRecommender:
    def __init__(self, rules_path="data/association_rules.csv", products_path="data/cleaned_dmart_dataset.csv"):
        self.rules_df = None
        self.products_df = None
        self.load_data(rules_path, products_path)
        
    def load_data(self, rules_path, products_path):
        if os.path.exists(rules_path):
            self.rules_df = pd.read_csv(rules_path)
        if os.path.exists(products_path):
            self.products_df = pd.read_csv(products_path)
            
    def get_all_products(self):
        """Returns sorted list of unique product names available for bundling."""
        if self.products_df is not None:
            return sorted(self.products_df["ProductName"].unique().tolist())
        return []
        
    def get_product_details(self, product_name):
        """Fetches product pricing, category, cost, and margin."""
        if self.products_df is None:
            return None
        sub = self.products_df[self.products_df["ProductName"] == product_name]
        if len(sub) == 0:
            return None
        row = sub.iloc[0]
        return {
            "name": row["ProductName"],
            "category": row["Category"],
            "price": float(row["SellingPrice"]),
            "cost": float(row["CostPrice"]),
            "unit_profit": float(row["UnitProfit"]),
            "margin_pct": float(row["ProfitMarginPct"])
        }
        
    def recommend(self, selected_product, top_k=3, w_lift=0.40, w_margin=0.40, w_confidence=0.20, bundle_discount_pct=5.0):
        """
        Generates both Traditional Apriori recommendations and Smart Profit-Aware recommendations.
        """
        if self.rules_df is None or len(self.rules_df) == 0:
            return None
            
        base_product = self.get_product_details(selected_product)
        if not base_product:
            return None
            
        # Filter rules where antecedent matches selected product
        # Matches exact name or compound antecedent starting with product
        candidate_rules = self.rules_df[
            (self.rules_df["Antecedent"] == selected_product) |
            (self.rules_df["Antecedent"].str.startswith(f"{selected_product} +"))
        ].copy()
        
        # If no direct match in antecedent, fallback to reverse association (Consequent == selected)
        if len(candidate_rules) == 0:
            reverse_rules = self.rules_df[self.rules_df["Consequent"] == selected_product].copy()
            if len(reverse_rules) > 0:
                # Swap antecedent and consequent
                candidate_rules = reverse_rules.rename(columns={
                    "Antecedent": "Consequent",
                    "Consequent": "Antecedent"
                })
                
        # If still none, fallback to same category high margin / high volume items
        if len(candidate_rules) == 0:
            cat = base_product["category"]
            same_cat = self.products_df[
                (self.products_df["Category"] == cat) & 
                (self.products_df["ProductName"] != selected_product)
            ].drop_duplicates(subset=["ProductName"]).copy()
            
            simulated_rules = []
            for _, r in same_cat.iterrows():
                simulated_rules.append({
                    "Antecedent": selected_product,
                    "Consequent": r["ProductName"],
                    "Support": 0.02,
                    "Confidence": 0.25,
                    "Lift": 1.25,
                    "Consequent_Category": r["Category"],
                    "Consequent_Price": r["SellingPrice"],
                    "Consequent_Profit": r["UnitProfit"],
                    "Consequent_Margin_Pct": r["ProfitMarginPct"],
                    "Bundle_Price": base_product["price"] + r["SellingPrice"],
                    "Bundle_Profit": base_product["unit_profit"] + r["UnitProfit"],
                    "Bundle_Margin_Pct": ((base_product["unit_profit"] + r["UnitProfit"]) / (base_product["price"] + r["SellingPrice"])) * 100
                })
            candidate_rules = pd.DataFrame(simulated_rules)
            
        if len(candidate_rules) == 0:
            return None
            
        # Deduplicate candidates on Consequent (keep highest lift)
        candidate_rules = candidate_rules.sort_values(by="Lift", ascending=False).drop_duplicates(subset=["Consequent"]).copy()
        # Remove self reference if any
        candidate_rules = candidate_rules[candidate_rules["Consequent"] != selected_product].copy()
        
        # 1. TRADITIONAL APRIORI (Frequency / Lift based only)
        traditional_candidates = candidate_rules.sort_values(by=["Lift", "Confidence"], ascending=False).head(top_k).copy()
        
        # 2. SMART PROFIT-AWARE RANKING (Multi-objective optimization function)
        # Min-Max Normalization
        lift_min, lift_max = candidate_rules["Lift"].min(), candidate_rules["Lift"].max()
        margin_min, margin_max = candidate_rules["Consequent_Margin_Pct"].min(), candidate_rules["Consequent_Margin_Pct"].max()
        
        lift_norm = (candidate_rules["Lift"] - lift_min) / (lift_max - lift_min) if lift_max > lift_min else 1.0
        margin_norm = (candidate_rules["Consequent_Margin_Pct"] - margin_min) / (margin_max - margin_min) if margin_max > margin_min else 1.0
        conf_norm = candidate_rules["Confidence"]
        
        candidate_rules["Hybrid_Score"] = (
            (w_lift * lift_norm) + 
            (w_margin * margin_norm) + 
            (w_confidence * conf_norm)
        ).round(4)
        
        smart_candidates = candidate_rules.sort_values(by="Hybrid_Score", ascending=False).head(top_k).copy()
        
        # 3. Bundle Aggregations & Financial Impact Analysis
        def compile_bundle(candidates_df, strategy_name):
            items = []
            total_items_price = base_product["price"]
            total_items_cost = base_product["cost"]
            total_items_profit = base_product["unit_profit"]
            
            for _, r in candidates_df.iterrows():
                items.append({
                    "name": r["Consequent"],
                    "price": float(r["Consequent_Price"]),
                    "profit": float(r["Consequent_Profit"]),
                    "margin_pct": float(r["Consequent_Margin_Pct"]),
                    "lift": float(r["Lift"]),
                    "confidence": float(r["Confidence"]),
                    "score": float(r.get("Hybrid_Score", r["Lift"]))
                })
                total_items_price += float(r["Consequent_Price"])
                total_items_profit += float(r["Consequent_Profit"])
                total_items_cost += (float(r["Consequent_Price"]) - float(r["Consequent_Profit"]))
                
            discount_val = round(total_items_price * (bundle_discount_pct / 100.0), 2)
            bundle_selling_price = round(total_items_price - discount_val, 2)
            bundle_net_profit = round(total_items_profit - discount_val, 2)
            bundle_margin_pct = round((bundle_net_profit / bundle_selling_price) * 100, 2) if bundle_selling_price > 0 else 0
            
            return {
                "strategy": strategy_name,
                "base_product": base_product,
                "recommended_items": items,
                "individual_sum_price": round(total_items_price, 2),
                "bundle_discount_amount": discount_val,
                "bundle_discount_pct": bundle_discount_pct,
                "bundle_price": bundle_selling_price,
                "bundle_profit": bundle_net_profit,
                "bundle_margin_pct": bundle_margin_pct
            }
            
        trad_bundle = compile_bundle(traditional_candidates, "Traditional Apriori (Frequency Only)")
        smart_bundle = compile_bundle(smart_candidates, "Smart Profit-Aware (Co-purchase + Margin)")
        
        profit_diff = smart_bundle["bundle_profit"] - trad_bundle["bundle_profit"]
        profit_growth_pct = round((profit_diff / trad_bundle["bundle_profit"] * 100), 1) if trad_bundle["bundle_profit"] > 0 else 0
        
        return {
            "selected_product": base_product,
            "traditional_bundle": trad_bundle,
            "smart_bundle": smart_bundle,
            "profit_difference": round(profit_diff, 2),
            "profit_growth_pct": profit_growth_pct,
            "explanation": self.generate_business_explanation(base_product, smart_bundle, trad_bundle, profit_diff, profit_growth_pct)
        }
        
    def generate_business_explanation(self, base, smart, trad, diff, growth):
        smart_names = [x["name"] for x in smart["recommended_items"]]
        trad_names = [x["name"] for x in trad["recommended_items"]]
        
        if diff > 0:
            msg = (
                f"💡 Business Advantage: Recommending high-margin cross-sell companions ({', '.join(smart_names)}) "
                f"instead of standard volume-only items ({', '.join(trad_names)}) delivers an extra ₹{diff:,.2f} profit "
                f"(+{growth}% profit expansion) per bundle sale while retaining strong purchase affinity."
            )
        else:
            msg = (
                f"💡 Business Insight: Both strategies recommend optimal items ({', '.join(smart_names)}). "
                f"The smart bundle achieves a balanced ₹{smart['bundle_profit']:,.2f} profit margin."
            )
        return msg

if __name__ == "__main__":
    rec = ProfitAwareRecommender()
    print("Testing Laptop Recommendation:")
    res = rec.recommend("Intel Core i5 Thin Laptop", top_k=3)
    if res:
        print("\n=== TRADITIONAL BUNDLE ===")
        print(f"Price: ₹{res['traditional_bundle']['bundle_price']}, Profit: ₹{res['traditional_bundle']['bundle_profit']}")
        print("\n=== SMART PROFIT-AWARE BUNDLE ===")
        print(f"Price: ₹{res['smart_bundle']['bundle_price']}, Profit: ₹{res['smart_bundle']['bundle_profit']}")
        print(f"Profit Advantage: +₹{res['profit_difference']} ({res['profit_growth_pct']}%)")
        print(f"\n{res['explanation']}")
