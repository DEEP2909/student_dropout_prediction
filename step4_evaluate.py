# =============================================================================
# STEP 4: EVALUATE THE MODEL
# =============================================================================
# Key metrics explained:
#
# Accuracy    = (TP + TN) / Total  →  overall correct predictions
# Precision   = TP / (TP + FP)     →  when we predict Dropout, how often right?
# Recall      = TP / (TP + FN)     →  of all actual Dropouts, how many caught?
# F1 Score    = 2 * (P*R)/(P+R)    →  balance of Precision and Recall
# ROC-AUC     = area under ROC curve → overall separability (1=perfect, 0.5=random)
#
# For this problem, RECALL is the most important metric.
# A missed dropout (FN) is worse than a false alarm (FP).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

with open("models/best_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc     = auc(fpr, tpr)

print(f"Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"ROC-AUC Score : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"],
            ax=axes[0], linewidths=1, linecolor="white", cbar=False)
# Annotate manually with labels + counts
labels_map = [["TN", "FP"], ["FN", "TP"]]
colors_map = [["#2ecc71", "#e74c3c"], ["#e74c3c", "#2ecc71"]]
for i in range(2):
    for j in range(2):
        axes[0].text(j + 0.5, i + 0.38, labels_map[i][j],
                     ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white")
        axes[0].text(j + 0.5, i + 0.62, str(cm[i, j]),
                     ha="center", va="center", fontsize=13, color="white")

axes[0].set_title(f"Confusion Matrix\nAccuracy: {acc*100:.1f}%",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# ROC Curve
axes[1].plot(fpr, tpr, color="#4C72B0", lw=2.5,
             label=f"Random Forest (AUC = {roc_auc:.2f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
axes[1].fill_between(fpr, tpr, alpha=0.12, color="#4C72B0")
axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Recall / Sensitivity)")
axes[1].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[1].legend(loc="lower right")

# Add threshold annotation
best_thresh_idx = np.argmax(tpr - fpr)
axes[1].plot(fpr[best_thresh_idx], tpr[best_thresh_idx],
             "ro", markersize=8, label="Best threshold")
axes[1].annotate("Optimal\nthreshold",
                 (fpr[best_thresh_idx], tpr[best_thresh_idx]),
                 textcoords="offset points", xytext=(15, -20), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="red"), color="red")

plt.suptitle("Step 4 — Model Evaluation", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/step4_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → plots/step4_evaluation.png")

metrics = classification_report(y_test, y_pred, target_names=["Graduate","Dropout"], output_dict=True)
metrics["roc_auc"] = roc_auc
metrics["accuracy"] = acc
with open("models/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved → models/eval_metrics.json")
