import transformers
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
from datasets import DatasetDict, load_from_disk
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from datetime import datetime

########################################
# Get the current date and time
########################################
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()

print('********************\nStarting execution\n********************\n')
print(f"Start date and time of execution: {formatted_time}\n\n")




#########################
# List of libraries
#########################
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Torch version: {torch.__version__}")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "true"






################################
# Load dataset
################################
# IMPORTANT:
# If embeddings are going to be computed from the exposure dataset, 
# you should comment out the encouragement dataset — and vice versa. 
# That means if the encouragement embeddings are going to be calculated, you should comment out the exposure-related code.
#########################


#########
## EXPOSURE
#########
path_dataset = "/home/lossioventuraj2/Code/07-Transcripts/script_detail/datasets/no_slide_windows/window_256"
dataset_name = "exposure"
dataset_file_path = f"{path_dataset}/{dataset_name}"
dataset = load_from_disk(dataset_file_path)

#########################
# ENCOURAGEMENT
#########################
path_dataset = "/home/lossioventuraj2/Code/07-Transcripts/script_detail/datasets/no_slide_windows/window_250"
dataset_name = "discourage_avoidance_merged_alt"
dataset_file_path = f"{path_dataset}/{dataset_name}"
dataset = load_from_disk(dataset_file_path)





######################################################
# Model Path and Names
######################################################
PATH_ = "meta-llama/"


LLAMA_MODEL_NAME = 'Llama-3.3-70B-Instruct'
LLAMA_MODEL_NAME_2 = 'Llama-3.1-8B-Instruct'
LLAMA_MODEL_NAME_3 = "Llama-3.1-8B"

# List of models to evaluate
MODEL_NAMES = [ 
    LLAMA_MODEL_NAME,
    LLAMA_MODEL_NAME_2,
    LLAMA_MODEL_NAME_3, 
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    "roberta-base", 
    "bert-base-uncased",
    "bert-large-uncased",
]


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device to use: {device}")




def extract_embeddings(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, return_tensors="pt", padding=True
    ).to(device)  # Ensure inputs are on the correct device

    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)

    word_ids = tokenized_inputs.word_ids(batch_index=0)  # Extract word IDs for alignment
    features, labels = [], []
    previous_word = None  # Track first subword of each word

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word:  # Take first subword embedding
            features.append(embeddings[0, i, :].cpu().numpy())  # Extract embedding
            labels.append(examples["ner_tags"][word_id])  # Store label
        previous_word = word_id

    return {"features": features, "labels": labels}


str_csv_files = "/home/lossioventuraj2/Code/07-Transcripts/script_detail/embeddings/"
str_embeddings = str_csv_files + dataset_name
path_embedding = Path(str_embeddings)
path_embedding.mkdir(parents=True, exist_ok=True)






for model_aux in MODEL_NAMES:
    print(f"\nProcessing model: {model_aux}")
    
    if model_aux in [LLAMA_MODEL_NAME, LLAMA_MODEL_NAME_2, LLAMA_MODEL_NAME_3]:
        model_name_or_path = PATH_ + model_aux
    else:
        model_name_or_path = model_aux
        

    if model_aux in ["mental/mental-roberta-base", "roberta-base"]: #model_aux == "roberta-base": #== "mental/mental-roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name_or_path, device_map='auto')
    dataset_with_embeddings = dataset.map(extract_embeddings, batched=False)
    
    X_train = np.vstack([embedding for sublist in dataset_with_embeddings["training"]["features"] for embedding in sublist])
    y_train = np.array([label for sublist in dataset_with_embeddings["training"]["labels"] for label in sublist])

    X_val = np.vstack([embedding for sublist in dataset_with_embeddings["validation"]["features"] for embedding in sublist])
    y_val = np.array([label for sublist in dataset_with_embeddings["validation"]["labels"] for label in sublist])

    X_test = np.vstack([embedding for sublist in dataset_with_embeddings["testing"]["features"] for embedding in sublist])
    y_test = np.array([label for sublist in dataset_with_embeddings["testing"]["labels"] for label in sublist])


    name_file_npy = model_aux.replace("/","_")
    print(f"\tSize training: {len(y_train)}")
    print(f"\tSize validation: {len(y_val)}")
    print(f"\tSize test: {len(y_test)}")


    np.save(path_embedding / f"X_train_{name_file_npy}.npy", X_train)
    np.save(path_embedding / f"X_val_{name_file_npy}.npy", X_val)
    np.save(path_embedding / f"X_test_{name_file_npy}.npy", X_test)

    np.save(path_embedding / f"y_train_{name_file_npy}.npy", y_train)
    np.save(path_embedding / f"y_val_{name_file_npy}.npy", y_val)
    np.save(path_embedding / f"y_test_{name_file_npy}.npy", y_test)



end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
execution_time_hours = execution_time_seconds / 3600

print('+++++++++++')
print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(f"Execution time: {execution_time_seconds:.2f} seconds")
print(f"Execution time: {execution_time_minutes:.2f} minutes")
print(f"Execution time: {execution_time_hours:.2f} hours")



print("\n\n*************************************\n\n")
print("\nProcessing complete.\n")
print("\n\n*************************************\n\n")
