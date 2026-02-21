# =============================================================================
# run_all.py  —  Run the entire project end-to-end
# =============================================================================
# Just run this one file and it will execute all 5 steps in order.
# Usage:  python run_all.py
# =============================================================================

import os, sys, subprocess, time

def run_step(step_file, step_name):
    print("\n" + "="*65)
    print(f"  {step_name}")
    print("="*65)
    start = time.time()
    result = subprocess.run([sys.executable, step_file], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌  {step_name} FAILED. Check errors above.")
        sys.exit(1)
    print(f"\n✅  {step_name} completed in {elapsed:.1f}s")

# Create required directories
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

total_start = time.time()

run_step("step1_load_explore.py",     "STEP 1: Load & Explore Data")
run_step("step2_preprocessing.py",   "STEP 2: Preprocessing (Imputation, Scaling, SMOTE)")
run_step("step3_train_model.py",     "STEP 3: Train Random Forest + Hyperparameter Tuning")
run_step("step4_evaluate.py",        "STEP 4: Evaluate Model (Accuracy, Confusion Matrix, ROC-AUC)")
run_step("step5_feature_importance.py", "STEP 5: Feature Importance + SHAP Values")

total = time.time() - total_start
print("\n" + "="*65)
print(f"  ALL STEPS COMPLETE  —  Total time: {total:.1f}s")
print("="*65)
print("\nOutput files:")
print("  📁 data/     → cleaned CSV, train/test arrays")
print("  📁 models/   → saved model (.pkl), eval metrics (.json)")
print("  📁 plots/    → all visualisations (.png)")
print("\nOpen any .png in the plots/ folder to see the results.")
