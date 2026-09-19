# Financial Transaction Anomaly Detection and Fraud Risk Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://streamlit.io/)
[![Academic Level](https://img.shields.io/badge/GTU-BE%20Computer%20Engineering%20Sem%207-green.svg)](https://www.gtu.ac.in/)
[![Internship](https://img.shields.io/badge/Internship-InfoLabz%20IT%20Services%20Pvt.%20Ltd.-purple.svg)](https://infolabz.in/)
[![Platform](https://img.shields.io/badge/Platform-FINSEC%20AI-cyan.svg)](#)

An enterprise-grade, end-to-end Machine Learning, Anomaly Detection, and Real-Time Risk Intelligence system designed to identify fraudulent transactions and abnormal behavioral patterns in financial mobile-money networks.

Developed as a Disciplinary / Online Internship Project for **Gujarat Technological University (GTU) BE Computer Engineering — Semester 7** in collaboration with **InfoLabz IT Services Pvt. Ltd.**

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Key Features](#4-key-features)
5. [Technology Stack](#5-technology-stack)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Dataset](#7-dataset)
8. [System Architecture](#8-system-architecture)
9. [Project Structure](#9-project-structure)
10. [Dashboard Pages](#10-dashboard-pages)
11. [Model Evaluation](#11-model-evaluation)
12. [How to Run Locally](#12-how-to-run-locally)
13. [Streamlit Deployment](#13-streamlit-deployment)
14. [Testing](#14-testing)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [Academic Context](#17-academic-context)

---

## 1. Project Overview
Digital financial transactions and mobile wallets process millions of high-velocity operations every second. **FINSEC AI** combines supervised ensemble classification with unsupervised outlier detection to safeguard financial ecosystems against fraudulent activity, account takeovers, and synthetic identity fraud.

Unlike rudimentary binary classifiers, this system incorporates:
- **Domain-Engineered Financial Signals**: Capturing balance discrepancies, account depletion patterns, and temporal off-peak activities.
- **Dual-Engine Detection**: Supervised probability estimation ($P(\text{Fraud})$) synthesized with unsupervised Isolation Forest anomaly scoring ($A_{\text{Score}}$).
- **Tri-Tier Risk Stratification**: Dynamically categorizing risk into `LOW`, `MEDIUM`, and `HIGH`.
- **Explainable Fraud Indicators**: Generating human-readable compliance justifications for non-technical faculty evaluators and bank compliance officers.

---

## 2. Problem Statement
Financial fraud incurs billions of dollars in annual global losses. Mobile financial services are particularly vulnerable to account takeover draining attacks via `TRANSFER` and `CASH_OUT` vectors. Conventional rule-based threshold systems fail due to high false-positive rates and rigidity against evolving attack patterns. Furthermore, naive machine learning models stumble over the **severe class imbalance** inherent in transaction data (~1% fraud), generating deceptively high accuracy while missing critical fraudulent events.

---

## 3. Objectives
1. Perform comprehensive Exploratory Data Analysis (EDA) on mobile money transaction ledgers.
2. We performed preprocessing and feature engineering using transaction amount, balance changes, transaction type and temporal patterns (deriving 23 domain features).
3. Train, benchmark, and evaluate multiple supervised models (Logistic Regression, Random Forest, XGBoost) and an unsupervised Isolation Forest.
4. Eliminate fabricated metrics by recording and reporting actual calculated evaluation scores (Precision, Recall, F1, ROC-AUC, PR-AUC).
5. Develop an intuitive, professional 7-page Streamlit web dashboard for compliance exploration and single-transaction real-time inference.

---

## 4. Key Features
- **Dual-Engine Risk Scoring**: Synthesizes supervised probability with unsupervised tree-isolation outlier scores.
- **23 Domain Financial Features**: Includes balance difference errors, 100% account drainage flags, and diurnal timing features.
- **7-Page Interactive Web Dashboard**: Dark FinTech glassmorphism aesthetic (`FINSEC AI`).
- **Complete Academic Artifacts**: Includes 10-chapter GTU report, full Jupyter notebook, system diagrams, and demo walkthrough script.

---

## 5. Technology Stack
- **Language**: Python 3.11+
- **Data Analytics**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Visualization**: Plotly (Dark Theme), Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Serialization**: Joblib
- **Testing**: Python standard `unittest`

---

## 6. Machine Learning Models

### Supervised Classification Models
1. **Logistic Regression (Baseline)**: Linear probabilistic classifier trained with balanced class weighting. High recall baseline.
2. **Random Forest Classifier (Selected Champion 🏆)**: Ensemble of 100 decorrelated decision trees using bagging and balanced subsampling. Delivers optimal F1-Score (93.80%) and PR-AUC (0.9814).
3. **XGBoost Classifier**: Gradient boosted decision trees sequentially optimizing residual gradients. Achieves top precision (92.97%).

### Unsupervised Anomaly Detection
- **Isolation Forest**: Isolates anomalies by constructing random feature partitions. Configured with a 3.0% contamination rate to catch unlabelled zero-day anomalies.

---

## 7. Dataset
The project uses a **PaySim-style synthetic financial transaction dataset**, modeled strictly after the mobile-money simulation benchmark by Lopez-Rojas et al.:
- **Total Transactions**: 60,000 verified records
- **Fraud Incidence**: 630 fraudulent transactions (1.05% realistic class imbalance)
- **Time Horizon**: 744 hourly time steps (31 days)
- **11 Canonical Attributes**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- **Integrity**: 0 missing values, 0 duplicate records, 0 negative amounts.

> *Note*: The dataset is generated synthetically to mirror authentic PaySim distributions without exposing private banking data.

---

## 8. System Architecture

```
Financial Transaction Dataset (PaySim Standard)
                      │
                      ▼
          Data Ingestion & Cleaning
       (Missing values, duplicates, types)
                      │
                      ▼
          Domain Feature Engineering
   (Balance discrepancies, draining flags, ratios)
                      │
           ┌──────────┴─────────────────────┐
           ▼                                ▼
 Supervised Classifiers            Unsupervised Anomaly
 (RF, XGBoost, LogReg)               (Isolation Forest)
           │                                │
           ▼                                ▼
    Fraud Probability                 Anomaly Score
           │                                │
           └──────────────┬─────────────────┘
                          ▼
            Composite Risk Assessment
              (LOW / MEDIUM / HIGH)
                          │
                          ▼
             Explainability Rationale
                          │
                          ▼
         Interactive Streamlit Dashboard
```

---

## 9. Project Structure

```
Financial-Fraud-Detection/
│
├── dashboard/
│   ├── app.py                    # Master 7-page Streamlit web application
│   ├── components.py             # UI KPI cards, badges, and dark Plotly charts
│   └── styles.py                 # Premium Dark FinTech CSS styling
│
├── dataset/
│   ├── generate_dataset.py       # High-fidelity PaySim benchmark generator
│   └── transactions.csv          # 60,000 verified transactions (4.8 MB)
│
├── models/
│   ├── fraud_model.pkl           # Best supervised classifier (Random Forest)
│   ├── anomaly_model.pkl         # Isolation Forest anomaly detector
│   ├── preprocessing.pkl         # Fitted StandardScaler & feature names
│   ├── evaluation_results.json   # Actual calculated performance metrics
│   ├── model_metadata.json       # Training metadata and hyperparameter records
│   └── anomaly_metadata.json     # Isolation Forest contamination statistics
│
├── notebooks/
│   ├── generate_notebook.py      # Notebook compilation script
│   └── fraud_detection_analysis.ipynb # 15-section executed academic Jupyter notebook
│
├── reports/
│   ├── GTU_Project_Report_Financial_Fraud_Detection.md # 10-Chapter GTU report
│   ├── demo_script.md            # Step-by-step 5-7 minute live demo script
│   └── diagrams/                 # Mermaid architecture & data flow diagrams
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── utils.py                  # Dynamic path resolvers and loggers
│   ├── data_preprocessing.py     # Schema validation, missing/duplicate audit
│   ├── feature_engineering.py    # 23 domain features & StandardScaler
│   ├── train_classification.py   # Stratified split, LR/RF/XGBoost training
│   ├── train_anomaly_model.py    # Isolation Forest unsupervised training
│   ├── evaluation.py             # Accuracy, Prec, Recall, F1, ROC/PR curves
│   └── prediction.py             # Dual-engine inference & explainability
│
├── tests/
│   ├── test_pipeline.py          # Automated ML pipeline unittest suite
│   ├── test_dashboard_pages.py   # Automated dashboard component unittests
│   └── verify_all_pages.py       # Headless end-to-end page verification
│
├── .gitignore                    # Clean Git ignore file
├── requirements.txt              # Production dependency specifications
├── README.md                     # Complete project documentation
├── PROJECT_STATUS.md             # Final project certification document
└── run.bat                       # One-click Windows execution launcher
```

---

## 10. Dashboard Pages
The interactive web dashboard includes 7 comprehensive pages:

1. **`01 Overview`**: Executive overview with 4 KPI cards, transaction volume breakdown donut chart, 24-hour diurnal activity vs. fraud profile, and recent ledger feed.
2. **`02 Transaction Analytics`**: Dynamic multi-parameter filtering, raw and log-scale amount distributions, and transaction volume trends.
3. **`03 Fraud Analysis`**: Channel concentration breakdown (100% fraud in `TRANSFER` and `CASH_OUT`), zero-balance account drain analysis, and suspicious transaction logs.
4. **`04 Anomaly Detection`**: Unsupervised Isolation Forest outlier scatter plot (`Transaction Amount` vs `Origin Balance Change`), anomaly score distributions, and contamination analysis.
5. **`05 Transaction Prediction`**: Live dual-engine inference interface with 3 academic demo presets, composite risk badges, and explainable diagnostics.
6. **`06 Model Performance`**: Empirical benchmark comparison table, confusion matrices, ROC curves, PR curves, and feature importance bar charts.
7. **`07 About Project`**: GTU BE Semester 7 academic specifications, InfoLabz internship credentials, pipeline architecture, and methodology documentation.

---

## 11. Model Evaluation

All metrics were computed on a held-out **20% Stratified Test Set (12,000 samples, 126 frauds)**:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Selection Rationale |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 98.10% | 35.34% | **97.62%** | 51.90% | 0.9986 | 0.9542 | Linear baseline; high recall but excessive false alarms |
| **Random Forest (Champion 🏆)** | **99.87%** | 91.67% | 96.03% | **93.80%** | **0.9996** | **0.9814** | **Top F1-Score & PR-AUC; optimal fraud capture** |
| **XGBoost Classifier** | **99.87%** | **92.97%** | 94.44% | 93.70% | 0.9993 | 0.9768 | Highly competitive gradient boosting ensemble |

### Confusion Matrix (Random Forest Test Set)
- **True Negatives (Legitimate Cleared)**: 11,863
- **False Positives (False Alarms)**: 11
- **False Negatives (Missed Frauds)**: 5
- **True Positives (Caught Frauds)**: 121

### Isolation Forest (Unsupervised)
- **Contamination Rate**: 3.00%
- **Anomalies Flagged**: 1,800 out of 60,000
- **Unsupervised Fraud Capture**: 224 / 630 frauds identified without any labels (35.56%)

---

## 12. How to Run Locally

### Prerequisites
- Python 3.11+ installed.

### Option A: One-Click Windows Launcher (Recommended)
Double-click `run.bat` in the project root folder.

### Option B: Command Line
```powershell
# 1. Clone or navigate to the project directory
cd Financial-Fraud-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 13. Streamlit Deployment

The project is fully structured for **Streamlit Community Cloud** deployment:
- **Main file path**: `dashboard/app.py`
- **Dependencies**: `requirements.txt` (located at repository root)
- **Relative Pathing**: All dataset and model paths resolve dynamically via `src/utils.py`.

### Steps to Deploy to Streamlit Cloud:
1. Push this repository to your GitHub account (`main` branch).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select your repository.
4. Set **Main file path** to `dashboard/app.py`.
5. Click **"Deploy!"**.

---

## 14. Testing

Run the automated unittest validation suite:

```powershell
# Run all 12 pipeline & dashboard tests
python -m unittest discover tests -v

# Run headless 7-page verification
python tests/verify_all_pages.py
```

---

## 15. Limitations
1. **Synthetic Dataset**: Trained on a PaySim-style synthetic benchmark. Live core banking systems include additional telemetry (card networks, IP geolocations, device fingerprints).
2. **Decision Support**: The platform serves as an investigative assistant for compliance analysts rather than autonomous transaction freezing.
3. **Graph Topology**: Does not currently execute multi-hop graph community detection for mule account rings.

---

## 16. Future Scope
1. **Streaming Data Pipeline**: Apache Kafka / Apache Spark streaming integration for sub-10ms latency scoring.
2. **Graph Neural Networks (GNN)**: Relational graph learning to uncover distributed money laundering networks.
3. **Dynamic SHAP Explanations**: Micro-level game-theoretic feature contribution force plots for each transaction.

---

## 17. Academic Context
- **Project Type**: GTU BE Computer Engineering Semester 7 — Disciplinary / Online Internship Project
- **Academic Year**: 2026
- **Internship Partner**: InfoLabz IT Services Pvt. Ltd., Ahmedabad
- **Domain**: Data Analytics & Machine Learning
- **Institution**: Gujarat Technological University (GTU)
