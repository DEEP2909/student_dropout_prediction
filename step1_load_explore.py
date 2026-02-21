# =============================================================================
# STEP 1: GENERATE & EXPLORE THE DATA
# =============================================================================
# In a real project you download the UCI dataset with:
#   from ucimlrepo import fetch_ucirepo
#   dataset = fetch_ucirepo(id=697)
#
# Here we generate realistic synthetic data with the same characteristics:
#   - 4,424 students, 35 features, 3:1 class imbalance (Graduate:Dropout)
#
# The features mimic the real UCI "Students' Dropout" dataset columns.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs("data",   exist_ok=True)
os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

N = 4424   # total students

# ── Generate realistic features ───────────────────────────────────────────────
def make_students(n, dropout=False):
    """Create synthetic student records. Dropout students have worse profiles."""
    d = 1 if dropout else 0  # offset to make dropouts look different
    return {
        "Age at enrollment"              : np.random.randint(17 + d*3, 35 + d*5, n),
        "Previous qualification grade"   : np.clip(np.random.normal(130 - d*15, 20, n), 60, 200),
        "Admission grade"                : np.clip(np.random.normal(125 - d*12, 22, n), 60, 200),
        "Curricular units 1st sem approved" : np.clip(np.random.poisson(5 - d*2, n), 0, 10),
        "Curricular units 1st sem grade"    : np.clip(np.random.normal(12 - d*3, 3, n), 0, 20),
        "Curricular units 2nd sem approved" : np.clip(np.random.poisson(5 - d*2, n), 0, 10),
        "Curricular units 2nd sem grade"    : np.clip(np.random.normal(12 - d*3, 3, n), 0, 20),
        "Curricular units 1st sem enrolled" : np.clip(np.random.poisson(6, n), 1, 10),
        "Curricular units 2nd sem enrolled" : np.clip(np.random.poisson(6, n), 1, 10),
        "Curricular units 1st sem evaluations": np.clip(np.random.poisson(8 - d, n), 0, 20),
        "Curricular units 2nd sem evaluations": np.clip(np.random.poisson(8 - d, n), 0, 20),
        "Tuition fees up to date"        : np.random.choice([0, 1], n, p=[0.1+d*0.25, 0.9-d*0.25]),
        "Scholarship holder"             : np.random.choice([0, 1], n, p=[0.7+d*0.1, 0.3-d*0.1]),
        "Debtor"                         : np.random.choice([0, 1], n, p=[0.85-d*0.2, 0.15+d*0.2]),
        "Displaced"                      : np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "Gender"                         : np.random.choice([0, 1], n),
        "International"                  : np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Application mode"               : np.random.randint(1, 18, n),
        "Application order"              : np.random.randint(0, 9, n),
        "Course"                         : np.random.randint(1, 18, n),
        "Daytime/evening attendance"     : np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "Previous qualification"         : np.random.randint(1, 17, n),
        "Nacionality"                    : np.random.randint(1, 21, n),
        "Mother's qualification"         : np.random.randint(1, 30, n),
        "Father's qualification"         : np.random.randint(1, 30, n),
        "Mother's occupation"            : np.random.randint(0, 46, n),
        "Father's occupation"            : np.random.randint(0, 46, n),
        "Marital status"                 : np.random.randint(1, 7, n),
        "Educational special needs"      : np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "GDP"                            : np.random.normal(1.5 - d*0.5, 2.5, n),
        "Inflation rate"                 : np.random.normal(1.0 + d*0.5, 1.5, n),
        "Unemployment rate"              : np.random.normal(11 + d*2, 3, n),
        "Curricular units 1st sem credited" : np.clip(np.random.poisson(0.5, n), 0, 5),
        "Curricular units 2nd sem credited" : np.clip(np.random.poisson(0.5, n), 0, 5),
        "Curricular units 1st sem without evaluations": np.clip(np.random.poisson(d, n), 0, 5),
    }

# 3:1 imbalance — 3318 graduates, 1106 dropouts
n_grad, n_drop = 3318, 1106
grad_data = make_students(n_grad, dropout=False)
drop_data = make_students(n_drop, dropout=True)

grad_df = pd.DataFrame(grad_data); grad_df["Target"] = 0
drop_df = pd.DataFrame(drop_data); drop_df["Target"] = 1

df = pd.concat([grad_df, drop_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape     : {df.shape}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"\nTarget distribution:")
print(df["Target"].value_counts().rename({0: "Graduate", 1: "Dropout"}))
ratio = n_grad / n_drop
print(f"\nClass imbalance ratio: {ratio:.1f}:1 (Graduate:Dropout)")

missing = df.isnull().sum().sum()
print(f"\nMissing values: {missing} (none — clean synthetic data)")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

counts = df["Target"].value_counts().sort_index()
bars = axes[0].bar(["Graduate", "Dropout"], counts.values,
                   color=["#4C72B0", "#DD8452"], edgecolor="black", width=0.5)
axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Students")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=11)
axes[0].set_ylim(0, max(counts.values) * 1.15)

axes[1].hist(df[df["Target"]==0]["Age at enrollment"], bins=25,
             alpha=0.65, label="Graduate", color="#4C72B0", edgecolor="white")
axes[1].hist(df[df["Target"]==1]["Age at enrollment"], bins=25,
             alpha=0.65, label="Dropout",  color="#DD8452", edgecolor="white")
axes[1].set_title("Age at Enrollment by Outcome", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Age"); axes[1].set_ylabel("Count"); axes[1].legend()

plt.suptitle("Step 1 — Exploratory Data Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/step1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved → plots/step1_eda.png")

df.to_csv("data/cleaned_data.csv", index=False)
print("Data saved → data/cleaned_data.csv")
