# 🤖 ML Apprentice Take-Home: Sentence Transformers & Multi-Task Learning

Welcome to my submission for the **ML Apprentice Take Home Exercise**. This project demonstrates my ability to build, train, and explain neural architectures with a focus on **Sentence Transformers** and **Multi-Task Learning (MTL)** using PyTorch and HuggingFace Transformers.

---

## 📄 Overview

This repository contains four tasks as described in the take-home prompt:

| Task | Description |
|------|-------------|
| **Task 1** | Implement a sentence transformer model to encode input sentences into fixed-length embeddings |
| **Task 2** | Expand the model to handle multi-task learning (sentence classification & sentiment analysis) |
| **Task 3** | Analyze different training configurations (frozen encoder, frozen head, etc.) |
| **Task 4** | Implement a multi-task training loop with hypothetical data and per-task metrics |

---

## Tasks Breakdown

### Task 1: Sentence Transformer Implementation
- Used three models: `all-MiniLM-L6-v2`, `paraphrase-mpnet-base-v2`, and `bert-base-nli-mean-tokens`
- Visualized embeddings using **PCA** and **t-SNE**
- Compared cosine similarity between sentence pairs
- 📍 [See: `Task_1.ipynb`](./Task_1.ipynb)

### Task 2: Multi-Task Learning Expansion
- Built a shared encoder with two classifier heads:
  - Task A: Sentence Classification
  - Task B: Sentiment Analysis
- Used synthetic data to test forward pass and independent loss calculations
- 📍 [See: `Task_2.ipynb`](./Task_2.ipynb)

### Task 3: Training Considerations
- Explored:
  - Freezing the entire model
  - Freezing only the transformer
  - Freezing only one classifier head
- Discussed use cases, trade-offs, and generalization potential

### Task 4: Training Loop (BONUS)
- Designed a training loop with alternating batches per task
- Evaluated with:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix
- Visualized loss curves across epochs

---

## Results Snapshot

- Perfect accuracy on both tasks with clean synthetic data
- Consistently decreasing loss curves
- Ideal confusion matrices due to well-separated data

---

## Setup & Reproducibility

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
