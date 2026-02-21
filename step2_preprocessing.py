# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================
# Raw data needs to be prepared before feeding into a model.
# Steps:
#   1. Separate features (X) and target (y)
#   2. Handle missing values via Imputation
#   3. Scale numeric features (StandardScaler)
#   4. Train/Test Split (80/20)
#   5. Fix class imbalance using SMOTE
#      (We implement a simple version without imblearn for compatibility;
#       on your machine: pip install imbalanced-learn and use SMOTE directly)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

df = pd.read_csv("data/cleaned_data.csv")

X = df.drop("Target", axis=1)
y = df["Target"]
feature_names = list(X.columns)

print(f"Features : {X.shape[1]}  |  Samples : {X.shape[0]}")

# ── 1. Impute missing values ──────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

# ── 2. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_names)

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"Train class balance (before SMOTE): {dict(y_train.value_counts())}")

# ── 4. SMOTE (manual implementation) ─────────────────────────────────────────
# On your machine install imblearn and replace this with:
#   from imblearn.over_sampling import SMOTE
#   X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
#
# What SMOTE does step by step:
#   a) Find all minority class (Dropout) samples
#   b) For each one, find its k nearest neighbours (also Dropout)
#   c) Pick a random neighbour, draw a random point on the line between them
#   d) That new synthetic point is added to training data
#   This creates realistic new samples, not just duplicates.

def simple_smote(X, y, random_state=42):
    """Oversample minority class to match majority class count."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)

    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    minority_mask = y_arr == 1
    X_min = X_arr[minority_mask]
    X_maj = X_arr[~minority_mask]

    n_to_generate = len(X_maj) - len(X_min)

    # Fit k-NN on minority samples
    nn = NearestNeighbors(n_neighbors=6)
    nn.fit(X_min)
    _, neighbours = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_to_generate):
        idx  = rng.randint(0, len(X_min))
        nn_idx = rng.choice(neighbours[idx][1:])   # skip self (index 0)
        gap  = rng.uniform(0, 1)
        new_point = X_min[idx] + gap * (X_min[nn_idx] - X_min[idx])
        synthetic.append(new_point)

    synthetic = np.array(synthetic)
    X_bal = np.vstack([X_arr, synthetic])
    y_bal = np.concatenate([y_arr, np.ones(n_to_generate, dtype=int)])

    # Shuffle
    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]

X_train_bal, y_train_bal = simple_smote(X_train, y_train)

print(f"\nAfter SMOTE:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

# ── Plot SMOTE effect ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, counts_arr, labels, title in [
    (axes[0], np.bincount(y_train.values), ["Graduate", "Dropout"], "Before SMOTE"),
    (axes[1], np.bincount(y_train_bal.astype(int)), ["Graduate", "Dropout"], "After SMOTE"),
]:
    bars = ax.bar(labels, counts_arr, color=["#4C72B0", "#DD8452"], edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts_arr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(counts_arr) * 1.15)

plt.suptitle("Step 2 — SMOTE Class Balancing", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/step2_smote.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved → plots/step2_smote.png")

# ── Save everything ───────────────────────────────────────────────────────────
np.save("data/X_train.npy", X_train_bal)
np.save("data/X_test.npy",  X_test.values)
np.save("data/y_train.npy", y_train_bal)
np.save("data/y_test.npy",  y_test.values)
pd.Series(feature_names).to_csv("data/feature_names.csv", index=False)
with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump({"imputer": imputer, "scaler": scaler, "features": feature_names}, f)
print("Preprocessed data saved.")
