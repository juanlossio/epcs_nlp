"""
Generate SBERT embeddings for multiple SentenceTransformer models across
train/validation/test subsets. Embeddings are chunk-averaged for long texts
using an overlap strategy (20% overlap).

Usage:
    python script.py <device_id> <start_index> <end_index>

Arguments:
    device_id   : CUDA device number to use
    start_index : Start index of model list (inclusive)
    end_index   : End index of model list (exclusive)

Workflow:
    1. Load text dataset (train/val/test).
    2. For each SBERT model in the provided slice:
           - Load model + tokenizer
           - Chunk text according to model max_seq_length
           - Compute embeddings for each chunk
           - Mean-pool chunk embeddings per document
           - Save embeddings to .npy files
    3. Print execution summary.

"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from sys import argv


# ============================================================
# Timestamp & Start Banner
# ============================================================

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()

print("********************")
print("Starting execution")
print("********************")
print(f"Start date/time: {current_time}\n\n")


# ============================================================
# Command-Line Arguments
# ============================================================

script, device_id, start_idx, end_idx = argv


# ============================================================
# Environment Information
# ============================================================

print("---- Environment Info ----")
print(f"NumPy version:        {np.__version__}")
print(f"Pandas version:       {pd.__version__}")
print(f"PyTorch version:      {torch.__version__}")
print("--------------------------\n")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("++++++++++++++++++++++++++++++\n")


# ============================================================
# Model List (SentenceTransformers)
# ============================================================

MODELS = [
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1",
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2",
    "msmarco-MiniLM-L6-cos-v5"
]

SUBSETS = ["2_training", "3_validation", "4_testing"]


# ============================================================
# Chunking Function
# ============================================================

def chunk_text(text: str, tokenizer, max_tokens: int, overlap: int):
    """
    Splits text into overlapping chunks based on model max token limit.

    Args:
        text (str): Input text.
        tokenizer: HuggingFace tokenizer.
        max_tokens (int): Maximum tokens per chunk.
        overlap (int): Overlap between consecutive chunks.

    Returns:
        List[str]: List of chunked text segments.
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    stride = max_tokens - overlap

    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


# ============================================================
# Embedding Function
# ============================================================

def get_embeddings(documents, model_name):
    """
    Computes mean-pooled document embeddings for a list of texts
    using the specified SentenceTransformer model.

    Args:
        documents (list[str]): List of input text samples.
        model_name (str): SBERT model name.

    Returns:
        np.ndarray: Array of document embeddings.
    """
    embeddings_all = []

    try:
        model = SentenceTransformer(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")

        max_len = model.get_max_seq_length()
        overlap = int(0.2 * max_len)

        for doc in documents:
            chunks = chunk_text(doc, tokenizer, max_len, overlap)

            chunk_vecs = model.encode(
                chunks,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False
            )

            doc_embedding = chunk_vecs.mean(axis=0)
            embeddings_all.append(doc_embedding)

        return np.array(embeddings_all)

    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return np.array([])


# ============================================================
# File Paths & Parameters
# ============================================================

# Path to the dataset directory.
# For **Exposure** tasks, change this to:  DATASET_PATH = "dataset_final/"
DATASET_PATH = "dataset_final_discavoid/"

# Directory where SBERT embeddings will be saved.
EMBEDDING_OUTPUT = "embeddings_sbert_discavoid/"

# Label column to use.
# For **Exposure** tasks, change this to:  LABEL_COLUMN = "Exposure"
LABEL_COLUMN = "DiscourageAvoidance"


# ============================================================
# Main Loop Over Models
# ============================================================

for model_name in MODELS[int(start_idx) : int(end_idx)]:

    print(f"\n----- Processing model: {model_name} -----")

    # --------------------
    # TRAIN
    # --------------------
    df_train = pd.read_csv(f"{DATASET_PATH}2_training.csv")
    print(f"2_training loaded: {df_train.shape}")

    X_train = get_embeddings(df_train["Text"].tolist(), model_name)
    y_train = df_train[LABEL_COLUMN].to_numpy()

    # --------------------
    # VALIDATION
    # --------------------
    df_val = pd.read_csv(f"{DATASET_PATH}3_validation.csv")
    print(f"3_validation loaded: {df_val.shape}")

    X_val = get_embeddings(df_val["Text"].tolist(), model_name)
    y_val = df_val[LABEL_COLUMN].to_numpy()

    # --------------------
    # TEST
    # --------------------
    df_test = pd.read_csv(f"{DATASET_PATH}4_testing.csv")
    print(f"4_testing loaded: {df_test.shape}")

    X_test = get_embeddings(df_test["Text"].tolist(), model_name)
    y_test = df_test[LABEL_COLUMN].to_numpy()

    # --------------------
    # SAVE EMBEDDINGS
    # --------------------
    safe_name = model_name.replace("/", "_")

    np.save(Path(EMBEDDING_OUTPUT) / f"X_train_{safe_name}.npy", X_train)
    np.save(Path(EMBEDDING_OUTPUT) / f"X_val_{safe_name}.npy", X_val)
    np.save(Path(EMBEDDING_OUTPUT) / f"X_test_{safe_name}.npy", X_test)

    np.save(Path(EMBEDDING_OUTPUT) / f"y_train_{safe_name}.npy", y_train)
    np.save(Path(EMBEDDING_OUTPUT) / f"y_val_{safe_name}.npy", y_val)
    np.save(Path(EMBEDDING_OUTPUT) / f"y_test_{safe_name}.npy", y_test)

    print(f"Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Saved embeddings for model: {safe_name}")


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
