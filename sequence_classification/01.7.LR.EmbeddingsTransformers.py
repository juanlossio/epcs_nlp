"""
============================================================
   Logistic Regression Benchmarks on Transformer-Based Model Embeddings
============================================================

This script evaluates multiple embedding models (LLMs and BERT family models)
by training a Logistic Regression classifier with extensive grid search.

Two classification tasks are supported:
    - Exposure
    - DiscourageAvoidance

Simply change the DATA_DIR and RESULTS_CSV variables to switch the task.

Pipeline:
---------
1. Load precomputed train/val/test embeddings for each model
2. Standardize embeddings
3. Grid-search Logistic Regression hyperparameters:
        - C values
        - solvers
        - class_weight (balanced / none)
4. Evaluate best configuration on train, validation, and test sets
5. Save full results to CSV

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

warnings.filterwarnings("ignore")  # cleaner output


# ============================================================
# Model Names (HuggingFace + Meta LLaMA)
# ============================================================

LLAMA_MODEL_NAME = 'Llama-3.3-70B-Instruct'
LLAMA_MODEL_NAME_2 = 'Llama-3.1-8B-Instruct'
LLAMA_MODEL_NAME_3 = "Llama-3.1-8B"

MODEL_NAMES = [
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    "roberta-base",
    "bert-base-uncased",
    "bert-large-uncased",
    LLAMA_MODEL_NAME_3,
    LLAMA_MODEL_NAME_2,
    LLAMA_MODEL_NAME,
]

# ============================================================
# Parameters
# ============================================================

SEEDS = [123]                  # Seeds for CV
CV_SPLITS = 5                  # 5-fold CV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'solver': ['liblinear'],    # stable with small embeddings
    'class_weight': [None, 'balanced'],
    'max_iter': [1000]
}

# ============================================================
# Choose Task Here:
#     - Exposure
#     - DiscourageAvoidance
# ============================================================

# ---- For DiscourageAvoidance ----
DATA_DIR = "embeddings_V1_discavoid/"
RESULTS_CSV = "final_LLM_results_discavoid.csv"

# ---- For Exposure ----
#DATA_DIR = "embeddings_V1_exposure/"
#RESULTS_CSV = "final_LLM_results_exposure.csv"


final_results = []
print("\n============================================================")
print("        LOGISTIC REGRESSION MODEL EVALUATION")
print(f"        Selected Task Directory: {DATA_DIR}")
print("============================================================\n")


# ============================================================
# Main evaluation loop
# ============================================================

for model_name in MODEL_NAMES:
    safe_name = model_name.replace("/", "_")
    print(f"\n--------------------------")
    print(f"Evaluating model: {safe_name}")
    print(f"--------------------------")

    # === Load embeddings and labels ===
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{safe_name}.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, f"X_val_{safe_name}.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, f"X_test_{safe_name}.npy"))

    y_train = np.load(os.path.join(DATA_DIR, f"y_train_{safe_name}.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, f"y_val_{safe_name}.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, f"y_test_{safe_name}.npy"))

    # === Standardize ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    all_seed_results = []

    # === CV for each seed ===
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
        df['seed'] = seed
        df['llm'] = safe_name
        df['params_str'] = df['params'].apply(str)
        all_seed_results.append(df)

    # === Aggregate CV results ===
    combined = pd.concat(all_seed_results, ignore_index=True)

    grouped = combined.groupby('params_str', as_index=False).agg({
        'mean_test_score': 'mean',
        'std_test_score': 'mean'
    })

    # Pick best hyperparameters
    top = grouped.sort_values(by='mean_test_score', ascending=False).head(1)

    # ============================================================
    # Train + Evaluate Best Model
    # ============================================================
    for _, row in top.iterrows():
        best_params = ast.literal_eval(row['params_str'])
        best_params.pop('max_iter', None)       # avoid duplication
        best_params.pop('random_state', None)

        clf_final = LogisticRegression(
            max_iter=1000,
            random_state=SEEDS[0],
            **best_params
        )
        clf_final.fit(X_train_scaled, y_train)

        val_auc   = roc_auc_score(y_val,   clf_final.predict_proba(X_val_scaled)[:, 1])
        test_auc  = roc_auc_score(y_test,  clf_final.predict_proba(X_test_scaled)[:, 1])

        final_results.append({
            'llm': safe_name,
            'params': best_params,
            'mean_cv_auc': row['mean_test_score'],
            'val_auc': val_auc,
            'test_auc': test_auc
        })

        # LaTeX-ready output
        print(f"{safe_name}\t & \t {test_auc:.4f}  &  ${best_params}$ \\\\")


# ============================================================
# Save Results
# ============================================================

df_final = pd.DataFrame(final_results)
df_final.to_csv(RESULTS_CSV, index=False)

print("\n============================================================")
print(f"Results saved to: {RESULTS_CSV}")
print("============================================================\n")
print(df_final.head())
