# FATE-X: Federated Adaptive Training and Explainable Intelligence Framework for Serverless Edge Computing

This repository contains the full MATLAB implementation of **FATE-X**, our proposed framework for federated adaptive learning, explainable modeling, and decentralized orchestration in serverless edge computing environments.

---

📂Repository Structure
FATE-X-Serverless/
│
├── Main.m                         # Main simulation script for FATE-X execution across multiple nodes
│
├── local_training.m               # Local training function performing model updates and explainability computation
├── federated_aggregation.m        # Aggregates local models into a global model using weighted averaging
├── predict_global.m               # Generates predictions using the final global model
├── partition_data.m               # Splits dataset into partitions assigned to serverless edge nodes
├── evaluate_metrics.m             # Computes Accuracy, Precision, Recall, F1-score, FPR, Communication Overhead, and Explainability Score
├── load_dataset.m                 # Loads the input dataset (e.g., CICAPT1_IIoTDataset2024.xlsx)
├── normalize_minmax.m             # Performs min–max feature normalization across samples
│
├── data/
│   └── CICAPT1_IIoTDataset2024.xlsx,E3_Darpa_20.xlsx, Wget.xlsx, Wget_Hour.xlsx, NSL_KDD.xlsx, StreamSpot.xlsx  # Input dataset used in all simulations (not included here for confidentiality)
│
├── results/
│   ├── accuracy_vs_nodes.png            # Accuracy vs. Number of Nodes
│   ├── precision_vs_nodes.png           # Precision vs. Number of Nodes
│   ├── recall_vs_nodes.png              # Recall vs. Number of Nodes
│   ├── f1score_vs_nodes.png             # F1-Score vs. Number of Nodes
│   ├── fpr_vs_nodes.png                 # False Positive Rate vs. Number of Nodes
│   ├── co_vs_nodes.png                  # Communication Overhead vs. Number of Nodes
│   └── es_vs_nodes.png                  # Explainability Score vs. Number of Nodes
│
├── README.md                      # Documentation file describing installation, usage, and results
└── .gitignore                     # Prevents temporary MATLAB or system files from being committed

