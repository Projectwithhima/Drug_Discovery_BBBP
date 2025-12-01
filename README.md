---

# 🧠 Drug Discovery – BBBP Prediction Pipeline

*A unified machine-learning, deep-learning, and graph-neural-network workflow for predicting blood–brain barrier permeability.*

---

## 📌 Overview

This repository contains a complete end-to-end pipeline for predicting **blood–brain barrier permeability (BBBP)** of small molecules.
The workflow integrates:

* RDKit-based molecular descriptor engineering
* Classical ML models
* Deep learning architectures
* Graph Neural Networks (GNNs)
* Large-scale virtual screening on **ChEMBL36**

The goal is to support **CNS drug discovery** by identifying compounds likely to penetrate the blood–brain barrier.

---

## 🎯 Problem Statement

**Predict whether a compound is BBB-permeable (BBB+) or non-permeable (BBB−).**
This helps prioritize CNS-active molecules during early drug-discovery stages.

---

## 📂 Datasets

### **1. BBBP Dataset**

* ~2,039 compounds
* Columns: *SMILES*, *p_np* (BBB+/BBB− label)

### **2. ChEMBL36 Dataset**

Used for external validation + large-scale virtual screening.

* Download:
  *[https://chembl.gitbook.io/chembl-interface-documentation/downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)*
* 4,082 **unlabeled** molecules curated as CNS-focused external dataset

---

## 🧪 Models Implemented

### **Traditional Machine Learning**

* Random Forest
* SVM (RBF kernel)
* XGBoost

### **Deep Learning**

* Dense Neural Network (DNN/MLP)
* 1D-CNN (on descriptor sequences)

### **Graph Neural Networks**

* **GCN** (PyTorch Geometric; molecular graph from SMILES)

---

## ⚙️ Methods

---

### **1. Molecular Descriptor Engineering**

RDKit converts SMILES → Mol → feature vectors.
Descriptors include:

#### **Physicochemical**

* Molecular Weight
* LogP
* TPSA
* HBD / HBA
* Rotatable bonds
* Aromatic ring count
* Fraction Csp3
* Heavy atom count
* Ring count
* Heteroatom count

#### **Drug-Likeness**

* QED
* Lipinski Rule Violations
* DrugLikeness Score

#### **CNS-Focused Metrics**

* **CNS_MPO_Score** (0–5 based on MW, LogP, TPSA, HBD, HBA)
* Molecular Flexibility
* Polar Surface Area Ratio

The function **`compute_comprehensive_descriptors()`** returns:

* Clean descriptor DataFrame
* Valid SMILES list

---

### **2. Data Preprocessing**

#### ✔ Outlier Removal

Using **IQR filtering** for all numeric descriptors:

[
x < Q1 - 1.5 \times IQR \quad \text{or} \quad x > Q3 + 1.5 \times IQR
]

Plots (not included here) typically show:

* Before/after outlier distribution
* Examples: MolWt, LogP

#### ✔ Class Imbalance (SMOTE)

* BBBP dataset is imbalanced (BBB+ >> BBB−)
* **SMOTE applied only to training split**

Visualizations may include:

* Class distribution before/after SMOTE
* PCA clusters showing synthetic samples

---

### **3. Model Training Pipeline**

#### 🧹 Steps:

1. Descriptor cleaning
2. Stratified train/test split (80/20, `random_state=42`)
3. Standardization (fit on train; transform test)
4. SMOTE oversampling (train only)

---

### 📊 Model Details

#### **Random Forest, SVM, XGBoost**

* Built via Scikit-Learn
* Hyperparameters optimized
* Evaluated on original test set

#### **DNN / MLP**

* Dense ReLU layers
* Dropout
* `Adam(lr=1e-3)`
* Early stopping

#### **1D-CNN**

* Conv1D + MaxPooling
* Input shape: `(n_samples, seq_len, 1)`
* Optimizer: Nadam
* Sigmoid final layer

#### **Graph Convolutional Network (GCN)**

* SMILES → graph (atom features, bond indices)
* GCNConv layers + global mean pooling
* Dense layer → sigmoid
* `Adam(lr=1e−3)` with BCE loss

### 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

---

## **4. External Screening on ChEMBL36**

### Descriptor-based models:

* Compute descriptors
* Scale using BBBP-trained StandardScaler
* Predict probabilities + binary class labels

### CNN:

* Reshape descriptors to `(n, seq_len, 1)`
* Predict BBB+ probability

### GCN:

* Convert SMILES → molecular graphs
* Predict BBB+ probability

All predictions collected into **`chembl_predictions`**, containing:

* SMILES
* Per-model predicted probability
* Per-model BBB+/BBB− label

This enables:

* Ranking of molecules
* Cross-model consensus scoring
* Selection of top CNS candidates

---

## ✔ Summary

This pipeline provides:

* A unified framework combining ML, DL, and GNN models
* A rich descriptor engineering module (RDKit-based)
* Robust preprocessing with IQR filtering + SMOTE
* A scalable system for large drug-discovery screening workflows
* Reproducible training and prediction steps for CNS permeability analysis

---

If you'd like, I can also generate:
📌 A shortened “Quick Start” version
📌 A diagram of the pipeline
📌 A section for installation & environment requirements
📌 Markdown badges (Python version, license, etc.)

Would you like any of these added?
