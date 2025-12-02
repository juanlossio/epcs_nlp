#######################################################
#######################################################
#######################################################
########### Fine Tuning Transformer-based Models
########### Exposure + Discourage avoidance 
########### Weighted Loss Function
########### Several models
#######################################################
#######################################################
#######################################################

from datasets import DatasetDict
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import numpy as np
from collections import Counter
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import classification_report
from transformers import TrainerCallback
from itertools import product
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from pathlib import Path
import pandas as pd
import os
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


########################################################
# MODELS (same as before)
########################################################

MY_HF_TOKEN = "[ADD YOUR TOKEN]"

MODEL_NAMES = [ 
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    "roberta-base", 
    "bert-base-uncased",
    "bert-large-uncased",
]

########################################################
# Hyperparameters
########################################################
param_grid = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 4e-4],
    "per_device_train_batch_size": [8, 16, 32, 64],
    "weight_decay": [0.01, 0.05, 0.1], 
    "num_train_epochs": [10]
}

param_combinations = list(product(*param_grid.values()))

########################################################
# DATASETS TO INCLUDE
########################################################

DATASETS = [
    ("exposure", ""),                      #  /window_256/exposure
    ("discourage_avoidance", "merged"),     #  /window_250/discourage_avoidance_merged
    ("discourage_avoidance", "alt"), 
    ("discourage_avoidance", "all"), 
    ("discourage_avoidance", "merged_alt") 
]


BASE_DATA_PATH = "/home/lossioventuraj2/Code/07-Transcripts/script_detail/datasets/no_slide_windows"

########################################################
# TOKENIZATION
########################################################

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, ner_labels in enumerate(examples["ner_tags"]):
        word_ids_after_tokenization = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_id_at in word_ids_after_tokenization:
            if word_id_at is None:
                label_ids.append(-100)
            elif word_id_at != previous_word_idx:
                label_ids.append(ner_labels[word_id_at])
            else:
                label_ids.append(-100)
            previous_word_idx = word_id_at
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

########################################################
# METRICS
########################################################

def compute_metrics_alt(p):
    predictions, labels = p
    shape_ = labels.shape
    probabilities = softmax(predictions, axis=2)
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p,l) in zip(prediction,label) if l != -100]
        for prediction,label in zip(predictions,labels)
    ]
    true_labels = [
        [label_list[l] for (p,l) in zip(prediction,label) if l != -100]
        for prediction,label in zip(predictions,labels)
    ]

    flat_predictions = [int(item) for sub in true_predictions for item in sub]
    flat_labels = [int(item) for sub in true_labels for item in sub]

    flat_probs = [
        prob[1]
        for probs, lbls in zip(probabilities, labels)
        for prob, l in zip(probs, lbls) if l != -100
    ]

    report = classification_report(flat_labels, flat_predictions, target_names=label_list, output_dict=True)

    try:
        auc_score = roc_auc_score(flat_labels, flat_probs) if len(set(flat_labels)) > 1 else float("nan")
    except ValueError:
        auc_score = float("nan")

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "epochs": num_train_epochs,
        "precision-0": report["0"]["precision"],
        "recall-0": report["0"]["recall"],
        "f-score-0": report["0"]["f1-score"],
        "precision-1": report["1"]["precision"],
        "recall-1": report["1"]["recall"],
        "f-score-1": report["1"]["f1-score"],
        "macro avg precision": report["macro avg"]["precision"],
        "macro avg recall": report["macro avg"]["recall"],
        "macro avg f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "Length": len(flat_labels),
        "Shape": shape_,
        "AUC": auc_score
    }

########################################################
# CALLBACK
########################################################

keep_key_details = ["learning_rate","batch_size","weight_decay","epochs","TrueLabels","Predictions","Probalities","AUC"]
remove_keys = ["TrueLabels","Predictions","Probalities"]

########################################################
# MAIN LOOP: BOTH DATASETS
########################################################

for feature_, pre_processing in DATASETS:

    # Resolve dataset path
    ds_suffix = feature_ if pre_processing == "" else f"{feature_}_{pre_processing}"
    WINDOW_SIZE = 256 if feature_ == "exposure" else 250
    
    dataset_path = f"{BASE_DATA_PATH}/window_{WINDOW_SIZE}/{ds_suffix}"

    print(f"\n\n==============================")
    print(f"Loading dataset: {dataset_path}")
    print("==============================\n")

    dataset = DatasetDict.load_from_disk(dataset_path)

    # IMPORTANT: Reset only once per dataset
    all_results = []
    
    ####################################################
    # Callback (needs access to lists but not resetting them wrongly)
    ####################################################
    class SaveEvalResultsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            epoch = state.epoch
            metrics["epoch"] = epoch
            aux_ = metrics.copy()
            # Store per-model metrics
            results.append({k.replace("eval_",""): v 
                            for k,v in aux_.items() if k.replace("eval_","") in keep_key_details})
            # Store global metrics
            aux_["model"] = model_name
            all_results.append({k.replace("eval_",""): v 
                                for k,v in aux_.items() if k.replace("eval_","") not in remove_keys})

    ####################################################
    # MODEL LOOP
    ####################################################
    for model_name in MODEL_NAMES:
        print(f"\n=== Evaluating Model: {model_name} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(tokenizer, (AutoTokenizer, RobertaTokenizerFast)):
            tokenizer.add_prefix_space = True

        tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        label_list = ["0", "1"]

        # Compute class weights
        train_labels = tokenized_dataset["training"]["labels"]
        all_labels = [l for sub in train_labels for l in sub if l != -100]
        count = Counter(all_labels)
        total = sum(count.values())
        weights = {label: total / c for label,c in count.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_tensor = torch.tensor([weights[i] for i in range(len(label_list))]).float().to(device)

        print("Class Weights:", weights)

        # IMPORTANT: Reset per-model results here
        results = []

        ################################################
        # HYPERPARAMETER LOOP
        ################################################
        for params in param_combinations:
            learning_rate, batch_size, weight_decay, num_train_epochs = params

            print(f"\n--- Hyperparameters lr={learning_rate}, batch={batch_size}, wd={weight_decay}, epochs={num_train_epochs}")

            training_args = TrainingArguments(
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=8,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                eval_strategy="epoch",
                seed=42,
                warmup_steps=500,
                max_grad_norm=1.0,
            )

            class WeightedTokenClassificationModel(torch.nn.Module):
                def __init__(self, model_name, num_labels, weights_tensor):
                    super().__init__()
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        model_name, num_labels=num_labels)
                    self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)

                def forward(self, input_ids, attention_mask=None, labels=None):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    logits = out.logits
                    if labels is not None:
                        loss = self.loss_fct(
                            logits.view(-1,len(label_list)), 
                            labels.view(-1)
                        )
                        return {"loss": loss, "logits": logits}
                    return {"logits": logits}

            def model_init():
                return WeightedTokenClassificationModel(model_name, len(label_list), weights_tensor)

            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=tokenized_dataset["training"],
                eval_dataset=tokenized_dataset["validation"],
                data_collator=data_collator,
                compute_metrics=compute_metrics_alt,
                callbacks=[SaveEvalResultsCallback()],
            )

            trainer.train()

        # Save per-model results
        out_dir = f"csv_results_{feature_}/FT_detail/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        df_detail = pd.DataFrame(results)
        df_detail["model"] = model_name
        df_detail.to_csv(
            f"{out_dir}/{ds_suffix}_{model_name.replace('/','_')}_results.csv",
            index=False
        )

    # Save combined results for all models and hyperparameters
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(
        f"csv_results_{feature_}/{ds_suffix}_ALL_MODELS.csv",
        index=False
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
