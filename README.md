BBBPermeator: BBBP Prediction & Drug Discovery Pipeline

A unified, reproducible pipeline to predict blood–brain barrier permeability (BBBP) of small molecules and enable CNS-relevant compound prioritization for drug discovery, using:

✅ Classical Machine Learning

✅ Deep Learning

✅ Graph Neural Networks (GCN)

🎯 Project Goal

Predict whether a compound is:

BBB-permeable (BBB+)

Non-permeable (BBB−)

to support faster design and screening of CNS drug-like molecules.

📊 Core Datasets
Dataset	Contents	Usage
BBBP (~2,039 compounds)	SMILES, p_np (BBB+/BBB− label)	Model training & evaluation
ChEMBL36 CNS Set (4,082 unlabeled molecules)	SMILES only	External virtual screening

Screening compounds from ChEMBL36 using trained ML, DL and GCN models to prioritize top candidates for CNS drug discovery.

🤖 Models Implemented
🔹 Traditional ML

Random Forest

SVM (RBF Kernel)

XGBoost

🔹 Deep Learning

Dense Neural Network (DNN / MLP)

1D-CNN on molecular descriptors

🔹 Graph Neural Network

Graph Convolutional Network (GCN) using PyTorch Geometric on SMILES-derived molecular graphs

🧬 Feature Engineering (Molecular Descriptors)

SMILES → Mol Objects using RDKit → descriptor calculation via:

⚗️ Physicochemical Features

MolWt, LogP, TPSA

H-Bond Donors/Acceptors

Rotatable & Aromatic Bonds, Rings

Heavy Atom, Heteroatom, Fraction Csp3

💊 Drug-Likeness

QED score

Lipinski rule violations

DrugLikeness score

🧠 CNS-Focused Scores

CNS_MPO_Score (0–5)

Polar Surface Area Ratio

Molecular Flexibility Index

⚙️ Data Preprocessing Pipeline

Outlier Removal

IQR filtering: removes extreme descriptor values

Visualized using before/after scatter plots

Class Balancing

SMOTE applied only on training data

Bar chart & PCA plots visualize synthetic sampling

Scaling

StandardScaler fitted on train, reused for all models & ChEMBL screening

Train/Test Split

Stratified 80/20, random_state=42

Evaluation Metrics

Accuracy, Precision, Recall, F1

ROC-AUC, Confusion Matrix

🔎 ChEMBL36 Virtual Screening

Descriptors generated using identical featurization

Scaled via original BBBP StandardScaler

GCN graphs built using the same atom/bond features

Models output:

Probability score (*_prediction)

Class (*_class) using threshold 0.5

Results stored in a consolidated table for ranking and intersection across models

📈 BBBP Test Results Summary
Model	Accuracy	F1 Score	AUC	Notes
XGBoost	~0.896	~0.938	~0.883	🏆 Best overall
Random Forest	~0.872	~0.923	—	Strong, interpretable
1D-CNN	~0.838	~0.900	~0.893	Highest AUC
SVM & DNN	~High precision	~Low recall	—	Conservative models
GCN	~0.830	~0.910	~0.740	Very high recall (1.0)
