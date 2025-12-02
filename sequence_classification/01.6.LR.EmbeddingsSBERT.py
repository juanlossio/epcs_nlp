"""
============================================================
     Logistic Regression Benchmarks on SBERT Embeddings
============================================================

This script evaluates multiple SBERT sentence transformer models
using a Logistic Regression classifier with hyperparameter
grid search, using precomputed embeddings for:

    • Exposure
    • DiscourageAvoidance

To switch tasks, modify:
    DATA_DIR  – path to embedding files
    RESULTS_CSV – output CSV name

Pipeline Overview:
------------------
1. Load train/val/test embeddings for each model
2. Standardize embeddings
3. Perform grid search on Logistic Regression:
       - C
       - class_weight (None, balanced)
       - solver
4. Select best mean CV AUC configuration
5. Train final model on full training set
6. Evaluate on train, validation, and test sets
7. Save all results to CSV

============================================================
"""

import os
import ast
import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")  # suppress verbose sklearn warnings


# ============================================================
# SBERT Model List
# ============================================================

MODEL_NAMES = [
    'all-mpnet-base-v2',
    'multi-qa-mpnet-base-dot-v1',
    'all-distilroberta-v1',
    'all-MiniLM-L12-v2',
    'multi-qa-distilbert-cos-v1',
    'all-MiniLM-L6-v2',
    'multi-qa-MiniLM-L6-cos-v1',
    'paraphrase-multilingual-mpnet-base-v2',
    'paraphrase-albert-small-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-MiniLM-L3-v2',
    'distiluse-base-multilingual-cased-v1',
    'distiluse-base-multilingual-cased-v2',
    'msmarco-MiniLM-L6-cos-v5'
]


# ============================================================
# Task Selection (Choose ONE)
# ============================================================

# ---- DiscourageAvoidance ----
DATA_DIR = "embeddings_sbert_discavoid/"
RESULTS_CSV = "results/final_SBERT_results_discavoid.csv"

# ---- Exposure ----
#DATA_DIR = "embeddings_sbert_exposure/"
#RESULTS_CSV = "results/final_SBERT_results_exposure.csv"


# ============================================================
# Parameters
# ============================================================

SEEDS = [42]                 # random seeds for robustness
CV_SPLITS = 5                # 5-fold cross-validation

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'solver': ['liblinear'],  # stable on small datasets + L2
    'class_weight': [None, 'balanced'],
    'max_iter': [1000]
}


# ============================================================
# Main Evaluation Loop
# ============================================================

os.makedirs("results", exist_ok=True)
final_results = []

print("\n============================================================")
print("          SBERT Logistic Regression Evaluation              ")
print("------------------------------------------------------------")
print(f" Task Embedding Directory: {DATA_DIR}")
print("============================================================\n")


for model_name in MODEL_NAMES:
    safe_name = model_name.replace("/", "_")
    print(f"\n---------- Evaluating: {safe_name} ----------")

    # === Load precomputed embeddings ===
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{safe_name}.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, f"X_val_{safe_name}.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, f"X_test_{safe_name}.npy"))

    y_train = np.load(os.path.join(DATA_DIR, f"y_train_{safe_name}.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, f"y_val_{safe_name}.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, f"y_test_{safe_name}.npy"))

    # === Standardization ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    all_seed_results = []

    # ============================================================
    # Cross-Validation
    # ============================================================

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=skf,
            n_jobs=-1,
            return_train_score=False
        )
        grid.fit(X_train_scaled, y_train)

        df = pd.DataFrame(grid.cv_results_)
        df["seed"] = seed
        df["model"] = safe_name
        df["params_str"] = df["params"].apply(str)  # easier grouping later

        all_seed_results.append(df)

    # Aggregate CV results across seeds
    combined = pd.concat(all_seed_results, ignore_index=True)

    grouped = combined.groupby("params_str", as_index=False).agg({
        "mean_test_score": "mean",
        "std_test_score": "mean"
    })

    # Best configuration
    top_cfg = grouped.sort_values(by="mean_test_score", ascending=False).head(1)

    # ============================================================
    # Final Training & Evaluation
    # ============================================================

    for _, row in top_cfg.iterrows():
        best_params = ast.literal_eval(row["params_str"])
        best_params.pop("max_iter", None)
        best_params.pop("random_state", None)

        clf_final = LogisticRegression(
            max_iter=1000,
            random_state=SEEDS[0],
            **best_params
        )
        clf_final.fit(X_train_scaled, y_train)

        val_auc   = roc_auc_score(y_val,   clf_final.predict_proba(X_val_scaled)[:, 1])
        test_auc  = roc_auc_score(y_test,  clf_final.predict_proba(X_test_scaled)[:, 1])

        final_results.append({
            "model": safe_name,
            "params": best_params,
            "mean_cv_auc": row["mean_test_score"],
            "val_auc": val_auc,
            "test_auc": test_auc
        })

        print(f"{safe_name} \t&\t {test_auc:.4f} \t&\t {best_params} \\\\")


# ============================================================
# Save Results
# ============================================================

df_results = pd.DataFrame(final_results)
df_results.to_csv(RESULTS_CSV, index=False)

print("\n============================================================")
print(f"Saved final SBERT results to: {RESULTS_CSV}")
print("============================================================\n")
print(df_results.head())
