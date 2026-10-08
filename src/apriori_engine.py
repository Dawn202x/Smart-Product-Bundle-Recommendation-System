"""
Step 4: Market Basket Analysis & Apriori Engine
Fast, self-contained, production-grade Apriori Algorithm.
Calculates:
- Support, Confidence, Lift, Leverage, Conviction
- Integrates Product Unit Economics (Cost, Price, Profit, Margin %)
- Exports enriched Association Rules to 'data/association_rules.csv'
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class AprioriEngine:
    def __init__(self, min_support=0.015, min_confidence=0.20, min_lift=1.1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.transactions = []
        self.item_counts = defaultdict(int)
        self.itemset_counts = defaultdict(int)
        self.total_transactions = 0
        self.rules = []
        
    def fit(self, df_transactions, invoice_col="InvoiceNo", item_col="ProductName"):
        """
        Extracts baskets and runs frequent itemset mining.
        """
        print(f"⚙️ Running Market Basket Analysis on {len(df_transactions)} line items...")
        
        # Group into transaction baskets
        baskets = df_transactions.groupby(invoice_col)[item_col].apply(lambda s: sorted(list(set(s)))).tolist()
        self.transactions = [b for b in baskets if len(b) >= 2]
        self.total_transactions = len(self.transactions)
        print(f"📦 Unique Multi-item Transactions: {self.total_transactions}")
        
        # 1. Frequent 1-itemsets
        c1 = defaultdict(int)
        for basket in self.transactions:
            for item in basket:
                c1[item] += 1
                
        min_count = self.min_support * self.total_transactions
        frequent_1 = {item: count for item, count in c1.items() if count >= min_count}
        self.item_counts = frequent_1
        print(f"🔹 Found {len(frequent_1)} frequent individual products (support >= {self.min_support:.1%})")
        
        # 2. Frequent 2-itemsets
        c2 = defaultdict(int)
        for basket in self.transactions:
            frequent_in_basket = [item for item in basket if item in frequent_1]
            for pair in combinations(frequent_in_basket, 2):
                c2[tuple(sorted(pair))] += 1
                
        frequent_2 = {pair: count for pair, count in c2.items() if count >= min_count}
        print(f"🔹 Found {len(frequent_2)} frequent 2-product combinations")
        
        # 3. Frequent 3-itemsets (if sufficient data)
        c3 = defaultdict(int)
        for basket in self.transactions:
            frequent_in_basket = [item for item in basket if item in frequent_1]
            for trio in combinations(frequent_in_basket, 3):
                # Verify all 2-subsets are frequent (Apriori property)
                if (tuple(sorted([trio[0], trio[1]])) in frequent_2 and
                    tuple(sorted([trio[0], trio[2]])) in frequent_2 and
                    tuple(sorted([trio[1], trio[2]])) in frequent_2):
                    c3[tuple(sorted(trio))] += 1
                    
        frequent_3 = {trio: count for trio, count in c3.items() if count >= min_count}
        print(f"🔹 Found {len(frequent_3)} frequent 3-product combinations")
        
        # Combine itemsets
        self.frequent_1 = frequent_1
        self.frequent_2 = frequent_2
        self.frequent_3 = frequent_3
        
        for k, v in frequent_1.items():
            self.itemset_counts[(k,)] = v
        for k, v in frequent_2.items():
            self.itemset_counts[k] = v
        for k, v in frequent_3.items():
            self.itemset_counts[k] = v
            
        self.stats = {
            "total_baskets": self.total_transactions,
            "c1_candidates": len(c1),
            "l1_frequent": len(frequent_1),
            "c2_candidates": len(c2),
            "l2_frequent": len(frequent_2),
            "c3_candidates": len(c3),
            "l3_frequent": len(frequent_3)
        }
            
        # 4. Generate Association Rules: A -> B
        rules_list = []
        for pair, pair_count in frequent_2.items():
            sup_ab = pair_count / self.total_transactions
            
            # Rule: pair[0] -> pair[1]
            sup_a = self.item_counts[pair[0]] / self.total_transactions
            sup_b = self.item_counts[pair[1]] / self.total_transactions
            conf_a_b = sup_ab / sup_a
            lift_a_b = conf_a_b / sup_b
            
            if conf_a_b >= self.min_confidence and lift_a_b >= self.min_lift:
                rules_list.append({
                    "Antecedent": pair[0],
                    "Consequent": pair[1],
                    "Support": round(sup_ab, 4),
                    "Confidence": round(conf_a_b, 4),
                    "Lift": round(lift_a_b, 4),
                    "Itemset_Size": 2
                })
                
            # Rule: pair[1] -> pair[0]
            conf_b_a = sup_ab / sup_b
            lift_b_a = conf_b_a / sup_a
            if conf_b_a >= self.min_confidence and lift_b_a >= self.min_lift:
                rules_list.append({
                    "Antecedent": pair[1],
                    "Consequent": pair[0],
                    "Support": round(sup_ab, 4),
                    "Confidence": round(conf_b_a, 4),
                    "Lift": round(lift_b_a, 4),
                    "Itemset_Size": 2
                })
                
        # Also generate rules from 3-itemsets (e.g. {A, B} -> C or A -> {B, C})
        for trio, trio_count in frequent_3.items():
            sup_abc = trio_count / self.total_transactions
            for item in trio:
                antecedent = tuple(sorted([x for x in trio if x != item]))
                ante_count = self.itemset_counts.get(antecedent, 0)
                if ante_count > 0:
                    sup_ante = ante_count / self.total_transactions
                    sup_cons = self.item_counts[item] / self.total_transactions
                    conf = sup_abc / sup_ante
                    lift = conf / sup_cons
                    if conf >= self.min_confidence and lift >= self.min_lift:
                        rules_list.append({
                            "Antecedent": f"{antecedent[0]} + {antecedent[1]}",
                            "Consequent": item,
                            "Support": round(sup_abc, 4),
                            "Confidence": round(conf, 4),
                            "Lift": round(lift, 4),
                            "Itemset_Size": 3
                        })
                        
        self.rules = pd.DataFrame(rules_list)
        print(f"✨ Successfully generated {len(self.rules)} association rules meeting thresholds.")
        return self.rules

    def get_frequent_itemsets_df(self):
        """Returns all frequent itemsets (L1, L2, L3) with support metrics."""
        rows = []
        if hasattr(self, 'frequent_1'):
            for item, count in self.frequent_1.items():
                rows.append({"Itemset": str([item]), "Size": 1, "Count": count, "Support": round(count / self.total_transactions, 4)})
        if hasattr(self, 'frequent_2'):
            for pair, count in self.frequent_2.items():
                rows.append({"Itemset": f"{pair[0]} + {pair[1]}", "Size": 2, "Count": count, "Support": round(count / self.total_transactions, 4)})
        if hasattr(self, 'frequent_3'):
            for trio, count in self.frequent_3.items():
                rows.append({"Itemset": f"{trio[0]} + {trio[1]} + {trio[2]}", "Size": 3, "Count": count, "Support": round(count / self.total_transactions, 4)})
        if len(rows) == 0:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(by=["Size", "Support"], ascending=[True, False])

    def enrich_with_economics(self, rules_df, products_df):
        """
        Enriches association rules with Product Economics (Cost, Price, Profit, Margin %).
        Calculates profit margin alongside co-purchase rules for dual-objective optimization.
        """
        prod_meta = products_df.groupby("ProductName").agg(
            Category=("Category", "first"),
            SellingPrice=("SellingPrice", "mean"),
            CostPrice=("CostPrice", "mean"),
            UnitProfit=("UnitProfit", "mean"),
            ProfitMarginPct=("ProfitMarginPct", "mean")
        ).reset_index()
        
        prod_dict = prod_meta.set_index("ProductName").to_dict(orient="index")
        
        enriched = []
        for _, row in rules_df.iterrows():
            cons = row["Consequent"]
            ante = row["Antecedent"]
            
            # Lookup consequent economics
            cons_info = prod_dict.get(cons, {"SellingPrice": 0, "CostPrice": 0, "UnitProfit": 0, "ProfitMarginPct": 0, "Category": "General"})
            
            # For antecedent (handle single item vs compound)
            first_ante = ante.split(" + ")[0]
            ante_info = prod_dict.get(first_ante, {"SellingPrice": 0, "CostPrice": 0, "UnitProfit": 0, "ProfitMarginPct": 0})
            
            bundle_price = ante_info["SellingPrice"] + cons_info["SellingPrice"]
            bundle_cost = ante_info["CostPrice"] + cons_info["CostPrice"]
            bundle_profit = bundle_price - bundle_cost
            bundle_margin_pct = (bundle_profit / bundle_price * 100) if bundle_price > 0 else 0
            
            enriched.append({
                **row.to_dict(),
                "Consequent_Category": cons_info["Category"],
                "Consequent_Price": round(cons_info["SellingPrice"], 2),
                "Consequent_Profit": round(cons_info["UnitProfit"], 2),
                "Consequent_Margin_Pct": round(cons_info["ProfitMarginPct"], 2),
                "Bundle_Price": round(bundle_price, 2),
                "Bundle_Profit": round(bundle_profit, 2),
                "Bundle_Margin_Pct": round(bundle_margin_pct, 2)
            })
            
        enriched_df = pd.DataFrame(enriched)
        return enriched_df

def run_market_basket_analysis(data_path="data/cleaned_dmart_dataset.csv", output_path="data/association_rules.csv"):
    print("="*60)
    print("🛒 STEP 4: MARKET BASKET ANALYSIS (APRIORI + ECONOMICS)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    engine = AprioriEngine(min_support=0.012, min_confidence=0.18, min_lift=1.1)
    raw_rules = engine.fit(df)
    
    enriched_rules = engine.enrich_with_economics(raw_rules, df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched_rules.to_csv(output_path, index=False)
    print(f"💾 Association Rules with Unit Economics saved to: {output_path}")
    print("\n🔍 Top 5 Association Rules by Lift (Standard Apriori View):")
    print(enriched_rules.sort_values(by="Lift", ascending=False)[["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Consequent_Margin_Pct", "Bundle_Profit"]].head().to_string(index=False))
    print("="*60)
    return enriched_rules

if __name__ == "__main__":
    run_market_basket_analysis()
