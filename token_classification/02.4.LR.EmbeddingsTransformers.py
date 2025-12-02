import os
import pandas as pd
import numpy as np
import sklearn
import multiprocessing
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import time
from datetime import datetime
import gc
gc.collect()

########################################
# Get the current date and time
########################################
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()

print("\n\n*************************************\n")
print("Start of Execution")
print("\n*************************************\n\n")

print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Sklearn version: {sklearn.__version__}")

print("\n****************\n")

total_cpus = multiprocessing.cpu_count()
n_jobs_value = max(1, int(total_cpus * 0.95))
print(f"Number of CPUs available: {total_cpus}")
print(f"Number of CPUs we will use: {n_jobs_value}")

####
## Change the feature as needed to calculate the logistic regression over the embeddings 
## for each of these features.
FEATURE_ = "exposure"  # or "discourage_avoidance"
DATASET_ = "discourage_avoidance_merged"

## Dataset names used in our work:
## - exposure
## - discourage_avoidance_merged / _all / _alt

global_parameters = {
    ## Input
    "feature": FEATURE_,
    "dataset_name": DATASET_,
    "embeddings": "embeddings/",

    ## Execution
    "random_state": 42,
    "k_fold": 5,
    "sampling": ["Undersampling", "Complete"],
    "score": "roc_auc",
    "no_best_models": 3,
    "subsets_eval": ["validation", "test"],

    ## Output
    "path_gridsearch": f"results_{FEATURE_}/LR/grid_search/",
    "path_val_test": f"results_{FEATURE_}/LR/val_test/",
    "individual_model_predictions": f"results_{FEATURE_}/LR/individual_model_predictions/",
}

print(f"Dataset: {global_parameters['dataset_name']}")

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [1000],
    "fit_intercept": [True],
    "class_weight": [None, "balanced"],
}

MODELS_ALL = [
    "Llama-3.1-8B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct",
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    "roberta-base",
    "bert-base-uncased",
    "bert-large-uncased",
]

path_embedding = Path(global_parameters["embeddings"]) / global_parameters["dataset_name"]

for sampling_training in global_parameters["sampling"]:
    all_MODELS_RES = []

    for MODEL_NAME in MODELS_ALL:

        MODEL_NAME = MODEL_NAME.replace("/", "_")

        print("\n++++++++\n")
        print(f"LLMs-based embeddings: {MODEL_NAME}\n--------")
        print(f"\tSampling training: {sampling_training}")

        X_train_initial = np.load(path_embedding / f"X_train_{MODEL_NAME}.npy")
        X_val_initial = np.load(path_embedding / f"X_val_{MODEL_NAME}.npy")
        X_test_initial = np.load(path_embedding / f"X_test_{MODEL_NAME}.npy")

        y_train_initial = np.load(path_embedding / f"y_train_{MODEL_NAME}.npy")
        y_val_initial = np.load(path_embedding / f"y_val_{MODEL_NAME}.npy")
        y_test_initial = np.load(path_embedding / f"y_test_{MODEL_NAME}.npy")

        print(f"X_train shape: {X_train_initial.shape}")
        print(f"X_val shape: {X_val_initial.shape}")
        print(f"X_test shape: {X_test_initial.shape}")

        print(f"y_train shape: {y_train_initial.shape}")
        print(f"y_val shape: {y_val_initial.shape}")
        print(f"y_test shape: {y_test_initial.shape}")

        ######
        ## Resampling
        ######
        if sampling_training == "Undersampling":
            print(f"Original class distribution: {Counter(y_train_initial)}")
            rus = RandomUnderSampler(random_state=global_parameters["random_state"])
            X_train, y_train = rus.fit_resample(X_train_initial, y_train_initial)
            print(f"Resampled class distribution: {Counter(y_train)}")

        elif sampling_training == "Complete":
            X_train, y_train = X_train_initial, y_train_initial
        ######

        print(f"Original class distribution: {Counter(y_train_initial)}")
        print(f"Original class distribution: {Counter(y_val_initial)}")
        print(f"Original class distribution: {Counter(y_test_initial)}")

        time.sleep(10)

        # Logistic regression
        clf = LogisticRegression()

        cv = StratifiedKFold(
            n_splits=global_parameters["k_fold"],
            shuffle=True,
            random_state=global_parameters["random_state"]
        )

        grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=cv,
            scoring=global_parameters["score"],
            verbose=1,
            n_jobs=n_jobs_value
        )

        grid_search.fit(X_train, y_train)

        # Save gridsearch results
        path_gridsearch = Path(global_parameters["path_gridsearch"])
        path_gridsearch.mkdir(parents=True, exist_ok=True)

        df_grid_results = pd.DataFrame(grid_search.cv_results_)
        df_grid_results.to_csv(
            path_gridsearch / f"SG.{sampling_training}.{global_parameters['dataset_name']}.{MODEL_NAME}.csv",
            index=False
        )

        print(f"Grid search results saved to '{path_gridsearch}/SG.{sampling_training}.{global_parameters['dataset_name']}.{MODEL_NAME}.csv'.")

        # Get top models
        sorted_results = df_grid_results.sort_values(by="mean_test_score", ascending=False)
        top_models = sorted_results.head(global_parameters["no_best_models"])

        all_model_results = []

        for idx in range(len(top_models)):
            row = top_models.iloc[idx]
            original_index = top_models.index[idx]

            print(f"\nEvaluating Model {idx + 1} with C={row['param_C']}, Solver={row['param_solver']}, Max_iter={row['param_max_iter']}")


            best_params = {
                "C": row["param_C"],
                "solver": row["param_solver"],
                "max_iter": row["param_max_iter"],
                "fit_intercept": row["param_fit_intercept"],
                "class_weight": row["param_class_weight"],  
            }


            

            model = LogisticRegression(**best_params)
            model.fit(X_train, y_train)

            for subset_ in global_parameters["subsets_eval"]:
                if subset_ == "validation":
                    X_val = X_val_initial
                    y_val = y_val_initial
                elif subset_ == "test":
                    X_val = X_test_initial
                    y_val = y_test_initial

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                y_proba_positive_class = y_proba[:, 1]
                roc_auc = roc_auc_score(y_val, y_proba_positive_class)

                class_report = classification_report(y_val, y_pred, output_dict=True)

                report_flat = {
                    "ModelName": MODEL_NAME,
                    "C": best_params["C"],
                    "solver": best_params["solver"],
                    "max_iter": best_params["max_iter"],
                    "fit_intercept": best_params["fit_intercept"],
                    "Subset": subset_,
                    "Original_IDX": original_index,
                    "IDX": idx,
                    "Model": f"Model_{idx + 1}",
                    "ROC_AUC": roc_auc,
                    "Accuracy": class_report["accuracy"],
                }

                for label, metrics in class_report.items():
                    if label != "accuracy":
                        for metric, value in metrics.items():
                            report_flat[f"{label}_{metric}"] = value

                all_model_results.append(report_flat)
                all_MODELS_RES.append(report_flat)

                df_predictions = pd.DataFrame({
                    "True_Label": y_val,
                    "Predicted_Label": y_pred,
                    "Probability_Class_0": y_proba[:, 0],
                    "Probability_Class_1": y_proba[:, 1]
                })

                individual_model_predictions = Path(
                    global_parameters["individual_model_predictions"]
                ) / subset_
                individual_model_predictions.mkdir(parents=True, exist_ok=True)

                df_predictions.to_csv(
                    individual_model_predictions / f"Pred.{sampling_training}.Model_{idx + 1}_{MODEL_NAME}.csv",
                    index=False
                )

                print(
                    f"Predictions saved: {individual_model_predictions}/Pred.{sampling_training}.Model_{idx + 1}_{MODEL_NAME}.csv"
                )

        path_val_test = Path(global_parameters["path_val_test"])
        path_val_test.mkdir(parents=True, exist_ok=True)

        df_model_results = pd.DataFrame(all_model_results)
        df_model_results.to_csv(
            path_val_test / f"Top_{global_parameters['no_best_models']}.{sampling_training}.{MODEL_NAME}_{global_parameters['dataset_name']}.csv",
            index=False
        )

        print(f"Top models saved to: {path_val_test}/Top_{global_parameters['no_best_models']}.{sampling_training}.{MODEL_NAME}_{global_parameters['dataset_name']}.csv")

    df_all_MODELS_RES = pd.DataFrame(all_MODELS_RES)
    df_all_MODELS_RES.to_csv(
        path_val_test / f"LLAMA-3.1.{sampling_training}.RESULTS_TOP_{global_parameters['no_best_models']}.csv",
        index=False
    )

    print(
        f"Saved: {path_val_test}/LLAMA-3.1.{sampling_training}.RESULTS_TOP_{global_parameters['no_best_models']}.csv"
    )


end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
execution_time_hours = execution_time_seconds / 3600

print("+++++++++++")
print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(f"Execution time: {execution_time_seconds:.2f} seconds")
print(f"Execution time: {execution_time_minutes:.2f} minutes")
print(f"Execution time: {execution_time_hours:.2f} hours")

print("\n\n*************************************\n")
print("End of Execution")
print("\n*************************************\n\n")