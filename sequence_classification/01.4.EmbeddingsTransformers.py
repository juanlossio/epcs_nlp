"""
Script for generating sentence/document embeddings using a collection of HuggingFace models.
The pipeline:
    1. Loads text datasets (train/val/test).
    2. Loads multiple transformer models and tokenizers.
    3. Chunks long documents (if needed) to respect token limits.
    4. Extracts embeddings by mean-pooling hidden states.
    5. Saves embeddings as .npy files for downstream classification tasks.
"""

import transformers
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from datetime import datetime


# ============================================================
# Environment and Version Information
# ============================================================

print("---- Environment Info ----")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Torch version: {torch.__version__}")
print("---------------------------\n")

# Ensure correct GPU ordering
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("++++++++++++++++++++++++++++++\n")


# ============================================================
# Timestamp Information
# ============================================================

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()

print("********************")
print("Starting execution")
print("********************")
print(f"Start time: {current_time}\n\n")


# ============================================================
# Model Paths and Names
# ============================================================

LLAMA_MODEL_NAME = "Llama-3.3-70B-Instruct"
LLAMA_MODEL_NAME_2 = "Llama-3.1-8B-Instruct"
LLAMA_MODEL_NAME_3 = "Llama-3.1-8B"
PATH_ = "meta-llama/"

MODEL_NAMES = [
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    "roberta-base",
    "bert-base-uncased",
    "bert-large-uncased",
    LLAMA_MODEL_NAME_2,
    LLAMA_MODEL_NAME_3,
    LLAMA_MODEL_NAME,
]

# Dataset subsets (CSV files)
SUBSETS = ["2_training", "3_validation", "4_testing"]


# ============================================================
# Text Chunking and Embedding Functions
# ============================================================

def chunk_text(text, max_tokens, overlap):
    """
    Splits a long document into overlapping chunks based on token count.
    
    Args:
        text (str): Input text.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
    
    Returns:
        List[str]: List of chunked text strings.
    """
    enc = tokenizer
    input_ids = enc.encode(text)
    chunks = []

    for i in range(0, len(input_ids), max_tokens - overlap):
        chunk_ids = input_ids[i : i + max_tokens]
        chunks.append(enc.decode(chunk_ids))

    return chunks


def get_embedding(text, max_tokens):
    """
    Computes the mean-pooled embedding for a given text input.

    Args:
        text (str): Input text for embedding.
        max_tokens (int): Token limit.

    Returns:
        np.ndarray: Embedding vector (1D).
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        embedding = embedding.cpu().numpy()[0]

    return embedding


# ============================================================
# Embeddings Output Paths and Parameters
# ============================================================

DATASET_DIR = "dataset_final_discavoid/"   # Use dataset_final for exposure
EMBEDDING_DIR = "embeddings_V1_discavoid/" # Use embeddings_final for exposure
LABEL_COLUMN = "DiscourageAvoidance"

Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)

OVERLAP = 100
DEFAULT_MAX_TOKENS = 512


# ============================================================
# Main Processing Loop Over Models
# ============================================================

for model_name in MODEL_NAMES:

    print(f"\n=== Processing model: {model_name} ===")

    # Determine correct path (local LLAMA models)
    if model_name in [LLAMA_MODEL_NAME, LLAMA_MODEL_NAME_2, LLAMA_MODEL_NAME_3]:
        model_path = PATH_ + model_name
        max_tokens = 128000
    else:
        model_path = model_name
        max_tokens = DEFAULT_MAX_TOKENS

    # Load tokenizer (with special handling for RoBERTa)
    special_tokenizer = model_name in ["mental/mental-roberta-base", "roberta-base"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_prefix_space=True if special_tokenizer else False
    )

    # Load base model for embedding extraction
    model = AutoModel.from_pretrained(model_path, device_map="auto")

    # ------------------------------
    # Load and Process Each Subset
    # ------------------------------

    for subset in SUBSETS:

        csv_path = os.path.join(DATASET_DIR, f"{subset}.csv")
        df = pd.read_csv(csv_path)

        print(f"{subset}: {df.shape}")

        embeddings_list = []

        for text in df["Text"].tolist():

            # Chunking only for non-LLAMA models
            if model_name not in [LLAMA_MODEL_NAME]:
                chunks = chunk_text(text, max_tokens, OVERLAP)
                chunk_embs = [get_embedding(chunk, max_tokens) for chunk in chunks]
                doc_emb = np.mean(chunk_embs, axis=0)
            else:
                doc_emb = get_embedding(text, max_tokens)

            embeddings_list.append(doc_emb.tolist())

        X = np.array(embeddings_list)
        y = np.array(df[LABEL_COLUMN].tolist())

        # Save .npy file
        name_safe = model_name.replace("/", "_")
        np.save(Path(EMBEDDING_DIR) / f"X_{subset}_{name_safe}.npy", X)
        np.save(Path(EMBEDDING_DIR) / f"y_{subset}_{name_safe}.npy", y)


# ============================================================
# Execution Summary
# ============================================================

end_time = time.time()
elapsed_sec = end_time - start_time

print("\n+++++++++++ Execution Summary +++++++++++")
print(f"Start time : {start_time}")
print(f"End time   : {end_time}")
print(f"Elapsed: {elapsed_sec:.2f} seconds")
print(f"Elapsed: {elapsed_sec/60:.2f} minutes")
print(f"Elapsed: {elapsed_sec/3600:.2f} hours")
print("+++++++++++++++++++++++++++++++++++++++++\n")

print("************** PROCESSING COMPLETE **************\n")
