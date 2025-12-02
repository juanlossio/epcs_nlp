# Automated Classification of Exposure and Encourage Events

### Code for: *“Automated Classification of Exposure and Encourage Events in Speech Data from Pediatric OCD Treatment”* (JAMIA Open)

This repository contains the implementation of **Automatic Coding of Transcriptions**, used in the JAMIA Open paper (Lossio-Ventura et al., 2025).

The workflow for detecting **Exposure Events** and **Encourage Approach** therapist behaviors from audio-recorded pediatric OCD treatment sessions consists of two major stages:

* **Step (A) Automatic Speech Recognition (ASR)** — Transcribing in-person therapy audio (Whisper used in the final system)
* **Step (B) Automatic Coding of Transcriptions** — Classifying *exposure* and *encourage* events from text

This repository provides all code for **Step (B): Automatic Coding of Transcriptions**, implemented using **sequence-level (fixed)** and **token-level (dynamic)** text segmentation.

---

## 📁 Repository Structure

```
epcs_nlp/
│
├── 01.dataset_preparation.ipynb
│
├── sequence_classification/
│   ├── 01.1.DatasetExposure.ipynb
│   ├── 01.2.DatasetDiscourage.ipynb
│   ├── 01.3.EmbeddingsSBERT.py
│   ├── 01.4.EmbeddingsTransformers.py
│   ├── 01.5.LR.BoW.TFIDF.py
│   ├── 01.6.LR.EmbeddingsSBERT.py
│   ├── 01.7.LR.EmbeddingsTransformers.py
│
├── token_classification/
│   ├── 02.1.DatasetExposure.ipynb
│   ├── 02.2.DatasetEncourage.ipynb
│   ├── 02.3.EmbeddingsTransformers.py
│   ├── 02.4.LR.EmbeddingsTransformers.py
│   ├── 02.5.FineTuning.py
│
└── README.md
```

---

# 1. Overview of the Coding Pipeline

## Step B — Automatic Coding of Transcriptions

The goal is to automatically classify therapist behaviors in session transcripts for two EPCS codes:

* **Exposure Event**
* **Encourage Approach**

Two complementary classification settings are provided:

* **Sequence-level classification** (fixed segments)
* **Token-level classification** (dynamic segments)

---

# 2. Sequence-Level Classification (Fixed Segments)

**Objective:** Predict whether an entire text segment contains an Exposure or Encourage event.
**Use case:** Coarse estimation of exposure presence or duration.

---

## 2.1 Dataset Preparation

**Files:**

* `sequence_classification/01.1.DatasetExposure.ipynb`
* `sequence_classification/01.2.DatasetDiscourage.ipynb`

These notebooks:

* Load aligned session transcripts
* Segment transcripts into **fixed-length text windows**
* Assign binary labels:

  * `0` = Non-Exposure / Non-Encourage
  * `1` = Exposure / Encourage
* Produce train/validation/test splits using site- and session-aware stratification

---

## 2.2 Embedding Generation

**Files:**

* `sequence_classification/01.3.EmbeddingsSBERT.py`
* `sequence_classification/01.4.EmbeddingsTransformers.py`

Supported embeddings:

### Classic NLP

* **Bag-of-Words (BoW)**
* **TF-IDF**
  (generated later in `01.5.LR.BoW.TFIDF.py`)

### Pretrained Transformer Embeddings

* **BERT / RoBERTa**
* **Sentence-BERT (SBERT)**
* **Domain-adapted models:** MentalBERT, MentalRoBERTa
* **Llama-3 family** (embedding extraction only)

Embeddings follow the method described in the paper (mean pooling of final hidden states).

---

## 2.3 Logistic Regression Classification

**Files:**

* `sequence_classification/01.5.LR.BoW.TFIDF.py`
* `sequence_classification/01.6.LR.EmbeddingsSBERT.py`
* `sequence_classification/01.7.LR.EmbeddingsTransformers.py`

These scripts implement:

* Logistic Regression with grid search
* `class_weight = [None, "balanced"]` included
* Support for multiple random seeds
* Evaluation metrics:

  * ROC-AUC
  * Accuracy

**Reported in the manuscript:**

* **Llama-3.3-70B-Instruct** achieved top AUC for Exposure (≈0.95)
* **MentalBERT** and **Llama-3.1-8B** performed best for Encourage classification

---

# 3. Token-Level Classification (Dynamic Segments)

**Objective:** Predict whether each **token** belongs to an Exposure or Encourage event.
This enables fine-grained, time-aligned dosage estimation.

---

## 3.1 Dataset Preparation

**Files:**

* `token_classification/02.1.DatasetExposure.ipynb`
* `token_classification/02.2.DatasetEncourage.ipynb`

These notebooks:

* Convert transcripts into **token-level labeled datasets**
* Align subword tokens using a first-subtoken labeling rule
* Merge short Encourage spans (<10 tokens) following paper methodology

---

## 3.2 Token Embedding Extraction

**File:**

* `token_classification/02.3.EmbeddingsTransformers.py`

Produces contextual token embeddings from:

* BERT / RoBERTa
* MentalBERT / MentalRoBERTa
* Llama-3 family (embedding extraction)

---

## 3.3 Logistic Regression (Token-Level)

**File:**

* `token_classification/02.4.LR.EmbeddingsTransformers.py`

Implements:

* Token-level logistic regression
* Class weighting and optional undersampling
* 5-fold cross-validation
* Evaluation with ROC-AUC and F1

Llama models were too large for fine-tuning, so only embedding-based LR was used.

---

## 3.4 Fine-Tuned Transformer Models

**File:**

* `token_classification/02.5.FineTuning.py`

Supports full fine-tuning of:

* BERT-base / BERT-large
* RoBERTa
* MentalBERT
* MentalRoBERTa

**Best results reported:**

* Exposure token classification: **AUC ≈ 0.85 (BERT-large)**
* Encourage token classification: **AUC ≈ 0.75**

Hyperparameters explored:

* Learning rate: `1e-5` to `4e-4`
* Batch sizes: 8, 16, 32, 64
* Epochs: 1–10
* Weighted cross-entropy

---

# 4. Running the Code

### Environment Requirements

```
Python >= 3.9
PyTorch >= 2.0
Transformers >= 4.x
Scikit-learn
NLTK                 # spaCy no longer required for preprocessing
SentenceTransformers
```

*(Updated: spaCy removed from preprocessing pipeline)*

Each script or notebook can be executed independently following the pipeline:

1. Dataset preparation
2. Embedding extraction
3. Logistic Regression or Fine-tuning

---

# 5. Citation

If you use this repository, please cite:

**Lossio-Ventura JA, Frank S, Ringlein G, et al.**
*Automated classification of exposure and encourage events in speech data from pediatric OCD treatment.*
**JAMIA Open**, 2025.

---

# 6. Contact

For questions or collaborations, contact:

**Juan Antonio Lossio-Ventura**
Machine Learning Core, NIMH/NIH
Email: **[juan.lossio@nih.gov](mailto:juan.lossio@nih.gov)**

