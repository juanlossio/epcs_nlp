"""
####################
ALTERNATIVE CODE that uses only NLTK, with spaCy removed from the preprocessing pipeline
####################

Baseline pipeline for text classification on Exposure / DiscourageAvoidance datasets.

This script performs:
1. Text preprocessing (tokenization, POS filtering, lemmatization/stemming)
2. Vectorization (BoW / TF-IDF with multiple n-gram ranges)
3. Feature scaling (StandardScaler on sparse matrices)
4. Logistic Regression with Grid Search + Stratified K-Fold CV
5. Evaluation on training, validation, and test sets
6. Result logging and export to CSV
7. Execution time measurement
"""

# ===============================================================
# Imports
# ===============================================================
import nltk
import pandas as pd
import numpy as np
import ast
import os
import time
from pathlib import Path
from datetime import datetime

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ===============================================================
# NLTK Models
# ===============================================================
# punkt = tokenizer
# averaged_perceptron_tagger_eng = English POS tagger for new NLTK versions
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# ===============================================================
# Execution Time Initialization
# ===============================================================
start_time = time.time()
start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("\n============================")
print("Starting execution")
print(f"Start time: {start_stamp}")
print("============================\n")


# ===============================================================
# Preprocessing Tools
# ===============================================================
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def get_wordnet_pos(tag: str):
    """
    Converts NLTK POS tag → WordNet POS tag.
    Defaults to NOUN.
    """
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN


def preprocess_text(text, mode='lemma', pos_filter=None):
    """
    Main preprocessing function.

    Args:
        text (str): Input text
        mode (str): 'original', 'lemma', or 'stem'
        pos_filter (str or None):
            - None     → keep all tokens
            - 'nva'    → nouns, verbs, adjectives
            - 'nouns'  → N only
            - 'verbs'  → V only
            - 'adjs'   → A only

    Returns:
        str: cleaned text
    """
    # Tokenize
    tokens = word_tokenize(text.lower())

    # POS-tag
    tagged = pos_tag(tokens)

    # POS filtering
    if pos_filter:
        allowed = {
            'nva': ['n', 'v', 'j'],
            'nouns': ['n'],
            'verbs': ['v'],
            'adjs': ['j']
        }[pos_filter]

        tagged = [(w, t) for w, t in tagged if t[0].lower() in allowed]

    # Apply stemming / lemmatization
    processed = []
    for word, tag in tagged:
        if mode == 'lemma':
            wn_tag = get_wordnet_pos(tag)
            processed.append(lemmatizer.lemmatize(word, pos=wn_tag))
        elif mode == 'stem':
            processed.append(stemmer.stem(word))
        else:
            processed.append(word)

    return ' '.join(processed)


# ===============================================================
# Dataset Configuration
# ===============================================================
DATA_DIR = "dataset_final"          # alternatives: "dataset_final_discavoid"
TEXT_COL = "Text"
LABEL_COL = "Exposure"              # or "DiscourageAvoidance"

train_df = pd.read_excel(f"{DATA_DIR}/2_training.xlsx")
val_df   = pd.read_excel(f"{DATA_DIR}/3_validation.xlsx")
test_df  = pd.read_excel(f"{DATA_DIR}/4_testing.xlsx")


# ===============================================================
# Experiment Configuration
# ===============================================================
SEEDS = [42]
CV_SPLITS = 5

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}

modes = ['original', 'lemma', 'stem']
pos_filters = [None, 'nva', 'nouns', 'verbs', 'adjs']

vectorizers = {
    'bow': CountVectorizer,
    'tfidf': TfidfVectorizer
}

ngram_ranges = [(1,1), (1,2), (1,3)]

final_results = []


# ===============================================================
# Main Loop: Preprocessing → Vectorization → CV → Evaluation
# ===============================================================
for mode in modes:
    for pos_filter in pos_filters:

        print(f"\n--- Preprocessing: mode={mode}, pos_filter={pos_filter} ---")

        # Apply preprocessing to all splits
        def apply_preprocessing(df):
            if mode == 'original' and pos_filter is None:
                return df[TEXT_COL].str.lower()
            return df[TEXT_COL].apply(lambda x: preprocess_text(x, mode, pos_filter))

        train_clean = apply_preprocessing(train_df)
        val_clean   = apply_preprocessing(val_df)
        test_clean  = apply_preprocessing(test_df)

        for vect_name, vect_cls in vectorizers.items():
            for ngram in ngram_ranges:

                setting_name = f"{mode}_{pos_filter or 'all'}_{vect_name}_{ngram}"
                print(f"→ Vectorizer={vect_name}, ngram={ngram}")

                # Vectorizer
                vect = vect_cls(ngram_range=ngram, max_features=5000)
                X_train = vect.fit_transform(train_clean)
                X_val   = vect.transform(val_clean)
                X_test  = vect.transform(test_clean)

                # Scale sparse matrix
                scaler = StandardScaler(with_mean=False)
                X_train_s = scaler.fit_transform(X_train)
                X_val_s   = scaler.transform(X_val)
                X_test_s  = scaler.transform(X_test)

                y_train = train_df[LABEL_COL].values
                y_val   = val_df[LABEL_COL].values
                y_test  = test_df[LABEL_COL].values

                all_seeds = []

                # Cross-validation with multiple seeds
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

                    grid.fit(X_train_s, y_train)

                    df_results = pd.DataFrame(grid.cv_results_)
                    df_results['seed'] = seed
                    df_results['setting'] = setting_name
                    df_results['params_str'] = df_results['params'].apply(str)
                    all_seeds.append(df_results)

                # Combine seeds
                combined = pd.concat(all_seeds, ignore_index=True)

                # Find best param set (mean across seeds)
                grouped = combined.groupby('params_str', as_index=False).agg({
                    'mean_test_score': 'mean',
                    'std_test_score': 'mean'
                })

                best_row = grouped.sort_values(by='mean_test_score', ascending=False).iloc[0]
                best_params = ast.literal_eval(best_row['params_str'])

                # Train final model using best params
                clf_final = LogisticRegression(
                    max_iter=1000,
                    random_state=SEEDS[0],
                    **best_params
                )
                clf_final.fit(X_train_s, y_train)

                # Compute metrics
                result = {
                    'setting': setting_name,
                    'params': best_params,
                    'mean_cv_auc': best_row['mean_test_score'],
                    'train_auc': roc_auc_score(y_train, clf_final.predict_proba(X_train_s)[:, 1]),
                    'val_auc':   roc_auc_score(y_val,   clf_final.predict_proba(X_val_s)[:, 1]),
                    'test_auc':  roc_auc_score(y_test,  clf_final.predict_proba(X_test_s)[:, 1])
                }

                final_results.append(result)


# ===============================================================
# Export Results
# ===============================================================
results_df = pd.DataFrame(final_results)
results_df = results_df.sort_values(by='mean_cv_auc', ascending=False)
results_df.to_csv(f"ALL_BASELINE_{LABEL_COL}.csv", index=False)

print("\nTop results:")
print(results_df.head(10))


# ===============================================================
# Execution Time Summary
# ===============================================================
end_time = time.time()
elapsed_sec = end_time - start_time

print("\n============================")
print("Execution complete")
print(f"Total time: {elapsed_sec:.2f} sec")
print(f"Total time: {elapsed_sec/60:.2f} min")
print(f"Total time: {elapsed_sec/3600:.2f} hrs")
print("============================\n")
