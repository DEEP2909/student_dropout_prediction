# =============================================================================
# STEP 3: TRAIN THE RANDOM FOREST MODEL
# =============================================================================
# A Random Forest works like this:
#   - Build 100s of Decision Trees, each trained on a random subset of data
#     and a random subset of features (this is called "bagging" + "feature randomness")
#   - To predict: every tree votes, majority wins
#
# Why this works: Each tree makes different errors. When you average them,
# the errors cancel out → the ensemble is more accurate than any single tree.
# This is called the "Wisdom of Crowds" principle.
#
# Hyperparameter Tuning with GridSearchCV:
#   Try every combination of settings, pick the best one via cross-validation.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle, time

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

print(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

# ── Baseline (default settings) ───────────────────────────────────────────────
print("\n--- Baseline Random Forest ---")
base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(base, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Grid Search ───────────────────────────────────────────────────────────────
# We try different combinations of:
#   n_estimators    : how many trees (more = better but slower)
#   max_depth       : max depth per tree (None = grow fully)
#   min_samples_leaf: min samples at a leaf (higher = simpler/less overfit)
param_grid = {
    "n_estimators"    : [100, 200],
    "max_depth"       : [None, 15],
    "min_samples_leaf": [1, 2],
}

print("\n--- Grid Search (trying all hyperparameter combinations) ---")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
t0 = time.time()
gs.fit(X_train, y_train)
print(f"\nGrid search done in {time.time()-t0:.1f}s")
print(f"Best params : {gs.best_params_}")
print(f"Best CV acc : {gs.best_score_:.4f}")

best = gs.best_estimator_

# ── Visualise CV results ──────────────────────────────────────────────────────
results = pd.DataFrame(gs.cv_results_)
results = results.sort_values("mean_test_score", ascending=False).head(8)

fig, ax = plt.subplots(figsize=(10, 5))
colors  = ["#2ecc71" if i == 0 else "#4C72B0" for i in range(len(results))]
bars    = ax.barh(range(len(results)),
                  results["mean_test_score"].values,
                  xerr=results["std_test_score"].values,
                  color=colors, edgecolor="black", capsize=4)

labels = [str(p) for p in results["params"].values]
ax.set_yticks(range(len(results)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Mean CV Accuracy")
ax.set_title("Grid Search Results — Top 8 Hyperparameter Combinations\n(Green = Best)",
             fontsize=12, fontweight="bold")
ax.set_xlim(results["mean_test_score"].min() - 0.01, 1.0)
for i, (bar, score) in enumerate(zip(bars, results["mean_test_score"].values)):
    ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
            f"{score:.4f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("plots/step3_gridsearch.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → plots/step3_gridsearch.png")

with open("models/best_random_forest.pkl", "wb") as f:
    pickle.dump(best, f)
print(f"Model saved → models/best_random_forest.pkl")
print(f"Train accuracy (sanity): {best.score(X_train, y_train):.4f}")
