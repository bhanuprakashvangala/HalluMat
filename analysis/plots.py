import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tabulate import tabulate

# -------- 1️⃣ Load Datasets -------- #
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

hallucination_summary = pd.read_csv(DATA_DIR / "Hallucination_Summary_per_Group.csv")
hallucination_distribution = pd.read_csv(DATA_DIR / "Hallucination_Level_Distribution_per_Group.csv")
hallumat_data = pd.read_csv(DATA_DIR / "HalluMatData.csv")

# -------- 2️⃣ Convert Hallucination Levels to Numeric -------- #
hallumat_data['Hallucination_Level'] = hallumat_data['Hallucination_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
hallumat_data['Computed_Hallucination_Level'] = hallumat_data['Computed_Hallucination_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})

# -------- 3️⃣ Ensure Group_ID is Numeric and Handle Ranges -------- #
hallumat_data['Group_ID'] = pd.to_numeric(hallumat_data['Group_ID'], errors='coerce')
hallumat_data.dropna(subset=['Group_ID'], inplace=True)
hallumat_data['Group_ID'] = hallumat_data['Group_ID'].astype(int)

# Bin Group_IDs into Ranges (e.g., 1-100, 101-200, ...)
bin_size = 100
bins = list(range(0, hallumat_data['Group_ID'].max() + bin_size, bin_size))
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
hallumat_data['Group_Range'] = pd.cut(hallumat_data['Group_ID'], bins=bins, labels=labels, right=False)

# -------- 📊 4️⃣ General Visualizations -------- #

# Distribution of Hallucination Levels
plt.figure(figsize=(10, 5))
sns.histplot(hallumat_data['Hallucination_Level'], bins=20, kde=True, color='blue')
plt.title("Distribution of Hallucination Levels")
plt.xlabel("Hallucination Level")
plt.ylabel("Frequency")
plt.savefig(FIG_DIR / "level_distribution.png", dpi=200, bbox_inches="tight"); plt.close()

# Boxplot: Hallucination Levels Across Group Ranges
plt.figure(figsize=(12, 6))
sns.boxplot(x="Group_Range", y="Hallucination_Level", data=hallumat_data)
plt.xticks(rotation=45, ha='right')
plt.title("Hallucination Level Spread Across Groups (Binned)")
plt.xlabel("Group ID Range")
plt.ylabel("Hallucination Level (1=Low, 3=High)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(FIG_DIR / "level_spread_by_group.png", dpi=200, bbox_inches="tight"); plt.close()

# Line Plot: Hallucination Trend by Group Range
plt.figure(figsize=(12, 6))
sns.pointplot(x="Group_Range", y="Hallucination_Level", data=hallumat_data, estimator="mean", errorbar=None)
plt.xticks(rotation=45, ha='right')
plt.title("Average Hallucination Trend Across Group Ranges")
plt.xlabel("Group ID Range")
plt.ylabel("Average Hallucination Level")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(FIG_DIR / "level_trend_by_group.png", dpi=200, bbox_inches="tight"); plt.close()

# -------- 📊 5️⃣ Statistical Analysis -------- #

# ANOVA Test: Checking if Hallucination Levels significantly differ across groups
anova_result = f_oneway(*[group["Hallucination_Level"].dropna() for _, group in hallumat_data.groupby("Group_ID")])

print("\n🔍 **ANOVA Test Results (Hallucination Variability Across Groups)** 🔍")
print(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.6f}")

# Compute Hallucination Variability per Group
hallucination_variability = hallumat_data.groupby("Group_ID")["Hallucination_Level"].std().reset_index()
hallucination_variability.columns = ["Group_ID", "PHCS_Score"]

# Select Top 10 Most Inconsistent Groups
top_inconsistent_groups = hallucination_variability.sort_values(by="PHCS_Score", ascending=False).head(10)

# T-Test: High vs Low Variability Groups
high_variability_groups = top_inconsistent_groups
low_variability_groups = hallucination_variability.sort_values(by="PHCS_Score", ascending=True).head(10)

t_test_result = ttest_ind(
    hallumat_data[hallumat_data["Group_ID"].isin(high_variability_groups["Group_ID"])]["Hallucination_Level"].dropna(),
    hallumat_data[hallumat_data["Group_ID"].isin(low_variability_groups["Group_ID"])]["Hallucination_Level"].dropna(),
    equal_var=False
)

print("\n🔍 **T-Test Results (High vs Low Inconsistency Groups)** 🔍")
print(f"T-statistic: {t_test_result.statistic:.4f}, p-value: {t_test_result.pvalue:.6f}")

# -------- 📊 6️⃣ Key Tables for Paper -------- #

# 1️⃣ Dataset Overview Table
dataset_overview = hallumat_data.describe().loc[['count', 'mean', 'std', 'min', 'max']]
print("\n📊 **Dataset Overview:**")
print(tabulate(dataset_overview, headers='keys', tablefmt='grid'))

# 2️⃣ Top 10 Inconsistent Groups Table
print("\n📊 **Top 10 Most Inconsistent Groups:**")
print(tabulate(top_inconsistent_groups, headers='keys', tablefmt='grid'))

# 3️⃣ Hallucination Level Breakdown Table
hallucination_counts = hallumat_data['Hallucination_Level'].value_counts().sort_index()
hallucination_table = pd.DataFrame({'Hallucination_Level': hallucination_counts.index, 'Count': hallucination_counts.values})
print("\n📊 **Hallucination Level Distribution:**")
print(tabulate(hallucination_table, headers='keys', tablefmt='grid'))

# 4️⃣ Agreement between the assigned and recomputed hallucination levels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
y_true = hallumat_data["Hallucination_Level"]
y_pred = hallumat_data["Computed_Hallucination_Level"]
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
performance_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
    "Value": [accuracy_score(y_true, y_pred), precision, recall, f1],
}).round(3)
print("\n📊 **Agreement Metrics:**")
print(tabulate(performance_metrics, headers='keys', tablefmt='grid'))

print("\n✅ All visualizations, statistical tests, and tables successfully generated!")
