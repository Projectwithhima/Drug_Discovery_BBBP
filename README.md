# Drug Discovery BBBP

This repository contains a complete pipeline for predicting blood–brain barrier permeability (BBBP) of small molecules using classical machine learning, deep learning, and graph neural networks. The project combines RDKit-based molecular descriptor engineering, robust preprocessing, and multiple models to screen CNS‑relevant compounds from the ChEMBL36 database.

Project Overview
Problem: Predict whether a compound is BBB‑permeable (BBB+) or non‑permeable (BBB−) to support CNS drug discovery.

Core datasets:

BBBP dataset (~2,039 compounds; p_np label, SMILES).

External CNS‑focused set from ChEMBL36 (4,082 unlabeled molecules) for virtual screening.

Models implemented:

Traditional ML: Random Forest, SVM (RBF), XGBoost.

Deep learning: Dense Neural Network (DNN/MLP), 1D‑CNN on descriptor sequences.

Graph model: Graph Convolutional Network (GCN) on SMILES‑derived molecular graphs.

Main goals:

Build reproducible, comparable BBBP models on descriptors and graphs.

Analyze which molecular properties drive BBB permeability.

Screen ChEMBL36 and prioritize candidate CNS‑active molecules.

Methods
1. Molecular Descriptors and Feature Engineering
SMILES strings are converted to RDKit Mol objects.

For each molecule, the following descriptors are computed:

Physicochemical: MolWt, LogP, TPSA, H‑bond donors/acceptors, rotatable bonds, aromatic rings, fraction Csp3, heavy‑atom count, ring count, heteroatom count.

Drug‑likeness: QED, Lipinski violations, DrugLikeness score.

CNS‑focused: CNS_MPO_Score (0–5, based on MW, LogP, TPSA, HBD, HBA), MolecularFlexibility, PolarSurfaceAreaRatio.

The function compute_comprehensive_descriptors() returns a clean descriptor DataFrame plus the corresponding valid SMILES.

2. Data Preprocessing
Outlier removal:

Interquartile Range (IQR) method applied to all numeric features.

Compounds outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR for any key descriptor are removed.

Before/after scatter plots (e.g., MolWt, LogP) illustrate removal of extreme values.

Class imbalance handling:

The BBBP dataset is imbalanced (BBB+ >> BBB−).

SMOTE is applied only on the training split to oversample the minority class (BBB− or BBB+, depending on encoding) without leaking synthetic samples into the test set.

Bar plots show class counts before and after SMOTE.

PCA (2D) visualizations display original vs SMOTE‑generated points in feature space.

3. Model Training
Training pipeline (BBBP):

Cleaned descriptors + p_np label.

Stratified train–test split (80/20, random_state=42).

Standardization with StandardScaler (fit on train, transform train/test).

SMOTE applied to the training set only.

Model training:

RandomForest, SVM (RBF), XGBoost (scikit‑learn).

DNN/MLP (Keras): multi‑layer dense ReLU + dropout, Adam (lr=1e‑3), binary cross‑entropy, early stopping on validation loss.

1D‑CNN (Keras): adaptive Conv1D + MaxPooling stacks on shape=(seq_len, 1), Nadam (lr=1e‑3), sigmoid output, early stopping.

GCN (PyTorch Geometric): SMILES → molecular graph (atom features + bond indices) → stacked GCNConv layers → global mean pooling → dense layers → sigmoid output, Adam (lr=1e‑3), BCE loss.

Evaluation on the original (unbalanced) test set using:

Accuracy, Precision, Recall, F1‑score, ROC‑AUC, confusion matrices.

4. External Screening on ChEMBL36
Descriptor‑based models (RF, SVM, XGBoost, DNN, 1D‑CNN):

ChEMBL descriptors are computed and scaled using the same StandardScaler fitted on BBBP training data.

CNN inputs are reshaped to (n_samples, seq_len, 1).

GCN:

ChEMBL SMILES are converted to graphs with the same atom/bond featurization as BBBP.

Each model outputs BBB+ probability (*_prediction) and a binary class label (*_class) using a 0.5 threshold.

A consolidated chembl_predictions table stores, for every molecule:

SMILES

Per‑model probability

Per‑model BBB+/BBB− class tag.

This table is used for ranking and intersecting top candidates across models.

Results (BBBP Test Set)
All models achieve test accuracies in the ~0.81–0.90 range.

XGBoost:

Accuracy ≈ 0.896, F1 ≈ 0.938, AUC ≈ 0.883 (best overall).

Random Forest:

Accuracy ≈ 0.872, F1 ≈ 0.923 (strong, interpretable baseline).

1D‑CNN:

Accuracy ≈ 0.838, F1 ≈ 0.900, highest AUC ≈ 0.893.

DNN and SVM:

High precision but lower recall (more conservative models).

GCN:

Accuracy ≈ 0.83, precision ≈ 0.83, recall ≈ 1.0, F1 ≈ 0.91, AUC ≈ 0.74 (very high sensitivity to BBB+ at cost of more false positives).

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

