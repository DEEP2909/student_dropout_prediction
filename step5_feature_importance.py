# =============================================================================
# STEP 5: FEATURE IMPORTANCE + SHAP-STYLE ANALYSIS
# =============================================================================
# Two ways to understand WHICH features drive predictions:
#
# 1. Built-in Impurity Importance (fast, slightly biased toward high-cardinality)
#    → Comes free with Random Forest
#
# 2. Permutation Importance (unbiased, slower)
#    → Shuffle one feature at a time, measure accuracy drop
#    → Big drop = feature was important
#    → This is the same principle as SHAP's global feature importance
#
# 3. Individual Explanation (SHAP-style waterfall)
#    → For ONE student, show which features pushed the prediction up/down
#    → We use the model's decision path in trees to compute this
#    → On your machine: pip install shap → use shap.TreeExplainer for full SHAP
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from sklearn.inspection import permutation_importance

with open("models/best_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
feature_names = pd.read_csv("data/feature_names.csv").iloc[:, 0].tolist()

TOP_N = 10

# ── 1. Built-in Impurity Importance ──────────────────────────────────────────
imp     = model.feature_importances_
idx     = np.argsort(imp)[::-1][:TOP_N]
top_names = [feature_names[i] for i in idx]
top_vals  = imp[idx]

print("Top 10 features (built-in importance):")
for r, (n, v) in enumerate(zip(top_names, top_vals), 1):
    print(f"  {r:2d}. {n:<45s} {v:.4f}")

# ── 2. Permutation Importance ─────────────────────────────────────────────────
print("\nComputing permutation importance (may take ~20s)...")
perm = permutation_importance(model, X_test, y_test,
                               n_repeats=10, random_state=42, n_jobs=-1)
perm_idx  = np.argsort(perm.importances_mean)[::-1][:TOP_N]
perm_names = [feature_names[i] for i in perm_idx]
perm_means = perm.importances_mean[perm_idx]
perm_stds  = perm.importances_std[perm_idx]
print("Done.")

# ── 3. Individual SHAP-style explanation ──────────────────────────────────────
# Use predict_proba differences to approximate feature contributions
# (on real project: shap.TreeExplainer gives exact SHAP values)
student_idx = 5   # pick a student predicted as Dropout
student     = X_test[[student_idx]]
base_prob   = model.predict_proba(X_test)[:, 1].mean()  # baseline: mean dropout prob
student_prob = model.predict_proba(student)[0, 1]

contributions = []
for fi in range(len(feature_names)):
    perturbed = student.copy()
    perturbed[0, fi] = 0.0   # zero out feature
    new_prob = model.predict_proba(perturbed)[0, 1]
    contributions.append(student_prob - new_prob)

contributions = np.array(contributions)
top8_idx = np.argsort(np.abs(contributions))[-8:]
print(f"\nStudent #{student_idx} — Predicted dropout probability: {student_prob:.2f}")
print(f"Model predicted: {'Dropout' if student_prob > 0.5 else 'Graduate'}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Built-in vs Permutation Importance
x     = np.arange(TOP_N)
width = 0.38
axes[0].barh(x + width/2, top_vals[::-1], width,
             color="#4C72B0", label="Built-in Impurity", edgecolor="black", alpha=0.85)

# Normalize perm importance to same scale for comparison
perm_norm = perm_means[::-1] / (perm_means.max() / top_vals.max())
axes[0].barh(x - width/2, perm_norm, width,
             color="#DD8452", label="Permutation (normalized)", edgecolor="black", alpha=0.85)

axes[0].set_yticks(x)
axes[0].set_yticklabels(top_names[::-1], fontsize=9)
axes[0].set_xlabel("Importance Score")
axes[0].set_title(f"Top {TOP_N} Feature Importances\n(Built-in vs Permutation)",
                  fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].axvline(x=0, color="black", linewidth=0.5)

# Plot 2: Individual Student Explanation (SHAP-style waterfall)
sorted_top8 = top8_idx[np.argsort(contributions[top8_idx])]
vals  = contributions[sorted_top8]
names = [feature_names[i] for i in sorted_top8]
colors_bar = ["#DD8452" if v > 0 else "#4C72B0" for v in vals]

axes[1].barh(range(8), vals, color=colors_bar, edgecolor="black")
axes[1].set_yticks(range(8))
axes[1].set_yticklabels(names, fontsize=9)
axes[1].axvline(x=0, color="black", linewidth=1)
axes[1].set_xlabel("Feature contribution to dropout probability")
axes[1].set_title(
    f"Why did model predict for Student #{student_idx}?\n"
    f"Dropout prob: {student_prob:.2f}  |  Baseline: {base_prob:.2f}",
    fontsize=11, fontweight="bold")
axes[1].legend(handles=[
    mpatches.Patch(color="#DD8452", label="↑ Increases dropout risk"),
    mpatches.Patch(color="#4C72B0", label="↓ Decreases dropout risk"),
], fontsize=9)

plt.suptitle("Step 5 — Feature Importance & Individual Explanation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/step5_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → plots/step5_feature_importance.png")

print("\nTop 8 drivers for Student #{}: ".format(student_idx))
for i in reversed(sorted_top8):
    direction = "↑ risk" if contributions[i] > 0 else "↓ risk"
    print(f"  {feature_names[i]:<45s}  contribution={contributions[i]:+.3f}  {direction}")
