# 🧠 Multi-Label Toxic Comment Detection  
## Transformer-Based Classification & Semantic Clustering

---

## 📌 Project Overview

This project addresses the challenge of multi-label toxic comment detection using a hybrid NLP framework combining:

- Supervised classification
- Unsupervised semantic clustering

The analysis evaluates statistical, deep learning, and transformer-based paradigms on the Jigsaw Toxic Comment dataset (200k+ comments).

---

## 🎯 Objectives

- Detect six toxicity categories simultaneously:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
- Address severe class imbalance
- Compare sparse, static, and contextual embeddings
- Discover hidden toxic micro-communities using clustering

---

## 📊 Dataset

- Jigsaw Toxic Comment Classification Dataset
- ~160k training comments
- Multi-label structure (6 categories)
- Highly imbalanced minority classes (<1%)

---

## 🧪 Classification Approaches

### 1️⃣ Statistical Baseline
- TF-IDF (1–2 grams, 100k features)
- LinearSVC (One-vs-Rest)
- Ridge Classifier
- Threshold optimization for F1 maximization

### 2️⃣ Deep Learning
- FastText 300D embeddings
- Text CNN architecture (multi-kernel 3/4/5)
- Focal Loss (γ=2) for imbalance handling
- Fine-grained threshold calibration

### 3️⃣ Transformer (State-of-the-Art)
- DistilBERT (fine-tuned)
- Focal Loss (γ=3)
- AdamW optimizer
- Contextual embeddings (768D)
- Achieved:
  - **Macro ROC-AUC: 0.9858**
  - **Macro F1-score: 0.62**

---

## 🔎 Unsupervised Semantic Clustering

### Statistical Clustering
- TF-IDF + Truncated SVD
- K-Means
- HDBSCAN

### Semantic Clustering
- SBERT embeddings (384D)
- UMAP dimensionality reduction
- HDBSCAN density clustering
- Achieved 94% cluster purity
- Identified high-risk micro-communities (e.g., targeted hate groups)

---

## 📈 Key Insights

- Contextual transformers significantly outperform static embeddings (+30% F1 over TextCNN).
- LinearSVC outperforms static deep learning baseline in some cases.
- Focal Loss effectively improves minority-class detection.
- Semantic clustering (SBERT + HDBSCAN) uncovers structured toxic subcultures missed by supervised models.

---

## 🛠 Technical Skills Demonstrated

- Advanced NLP preprocessing
- Multi-label classification
- Transformer fine-tuning
- Focal Loss implementation
- Threshold calibration
- Model evaluation (ROC-AUC, F1, NMI, ARI, Silhouette)
- Semantic embedding analysis
- Density-based clustering
- Handling imbalanced datasets

---

## 🚀 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- FastText
- SBERT
- UMAP
- HDBSCAN
- NLTK
- Pandas
- NumPy
