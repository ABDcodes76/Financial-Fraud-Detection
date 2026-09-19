# GUJARAT TECHNOLOGICAL UNIVERSITY (GTU)
## BACHELOR OF ENGINEERING (COMPUTER ENGINEERING)
### FINAL YEAR PROJECT REPORT (SEMESTER 7)

---

# Financial Transaction Anomaly Detection and Fraud Risk Prediction System using Machine Learning

**Academic Degree:** Bachelor of Engineering in Computer Engineering  
**Academic Semester:** Semester 7 (Final Year)  
**University:** Gujarat Technological University (GTU), Ahmedabad, Gujarat  
**Project Category:** Disciplinary Project / Internship Project  
**Internship Organization:** InfoLabz IT Services Pvt. Ltd., Ahmedabad  
**Domain:** Data Analytics & Applied Machine Learning  
**Development Framework:** Python 3.11, Scikit-Learn, XGBoost, Streamlit, Plotly  

---

## EXECUTIVE ABSTRACT
Mobile financial systems, digital payment channels, and electronic wallets have experienced explosive growth globally. However, this growth has been accompanied by increasingly sophisticated fraudulent schemes, most notably unauthorized account takeovers and rapid funds siphonage. Traditional rule-based fraud detection systems suffer from high false-positive rates, rigid parameter thresholds, and complete vulnerability to novel attack patterns.

This project presents an end-to-end Machine Learning and Anomaly Detection framework tailored for financial transaction surveillance. Leveraging a PaySim-style synthetic financial transaction dataset modeled after the mobile money simulation benchmark (Lopez-Rojas et al.), the system integrates data preprocessing, domain-specific feature engineering (balance discrepancy modeling, account draining flags, temporal cyclical signals), supervised classification (Logistic Regression, Random Forest, XGBoost), and unsupervised outlier detection (Isolation Forest). 

Because financial fraud suffers from extreme class imbalance (~1.05% fraud in our 60,000-transaction benchmark), conventional evaluation metrics such as overall accuracy are mathematically deceptive. Consequently, models are rigorously evaluated using Precision-Recall Area Under the Curve (PR-AUC), F1-Score, and Recall on a 20% stratified holdout dataset. The trained Random Forest classifier achieved a top F1-Score of **93.80%**, a Recall of **96.03%**, and a PR-AUC of **0.9814**, outperforming linear baselines. Furthermore, the unsupervised Isolation Forest effectively flags novel anomalies with a 3.0% contamination prior, capturing 35.56% of fraudulent behaviors without supervision. A responsive, 7-page interactive Streamlit dashboard enables compliance officers to conduct real-time single-transaction risk scoring (LOW, MEDIUM, HIGH) backed by explainable diagnostic indicators.

---

## 1. INTRODUCTION & PROBLEM STATEMENT

### 1.1 Background
The digital financial ecosystem processes billions of dollars daily through automated payment gateways and mobile money platforms. Unlike legacy banking, digital transfers settle within milliseconds, leaving negligible time for manual compliance intervention.

### 1.2 Problem Statement
Financial fraud detection systems face three primary bottlenecks:
1. **Extreme Class Imbalance:** Fraudulent transactions represent less than 1% to 2% of total transaction volume. Naive machine learning models achieve >98% accuracy simply by classifying every transaction as legitimate, missing all malicious acts.
2. **Dynamic Attack Vectors:** Rule-based heuristics (e.g., flagging transactions strictly > $200,000) are easily circumvented by fraudsters splitting payments into sub-threshold tranches.
3. **Black-Box Opacity:** Complex machine learning models lack interpretability, preventing compliance analysts from understanding why a transaction was flagged.

### 1.3 Project Objectives
- Ingest and audit 60,000 realistic mobile money transactions adhering to the PaySim-style synthetic benchmark.
- Engineer domain features highlighting account draining, balance discrepancies, and temporal off-peak activity.
- Implement supervised classification using class-weighting and gradient boosting.
- Implement unsupervised Isolation Forest to detect behavioral anomalies.
- Synthesize supervised probability and unsupervised anomaly scores into a tri-tier risk framework (LOW, MEDIUM, HIGH).
- Provide human-interpretable explainability indicators for every transaction prediction.
- Deliver an interactive 7-page Streamlit web dashboard for compliance workflows.

---

## 2. LITERATURE SURVEY & COMPARATIVE ANALYSIS

| Feature / Dimension | Legacy Rule-Based Systems | Single Supervised ML Model | Proposed FraudGuard ML System |
|---|---|---|---|
| **Detection Methodology** | Hardcoded if-else threshold rules | Supervised binary classification | Dual-Engine: Supervised + Isolation Forest Anomaly |
| **Adaptability to New Fraud** | Zero (Requires manual rule updates) | Poor (Fails on out-of-distribution vectors) | High (Isolation Forest catches rare anomalies) |
| **Class Imbalance Handling** | Not applicable | Often ignored (High false negatives) | Stratified splitting, class weights, PR-AUC tuning |
| **Explainability** | High (Rule triggered is known) | Low (Black-box tree/neural output) | High (Rule-based explainability engine + feature importances) |
| **Risk Stratification** | Binary (Allowed / Blocked) | Binary or raw probability | Tri-Tier (LOW / MEDIUM / HIGH) with anomaly overlay |
| **Compliance Dashboard** | Static database queries | Jupyter notebook scripts | Interactive 7-page Streamlit analytical dashboard |

---

## 3. SYSTEM ARCHITECTURE & METHODOLOGY

The system architecture is structured across four decoupled layers:

```
[ Financial Transaction Stream / CSV ]
                │
                ▼
   [ Data Ingestion & Sanitization ]
   (Missing value checks, duplicate removal, non-negative clipping)
                │
                ▼
   [ Domain Feature Engineering ]
   (Balance discrepancies, draining flags, log amount, temporal hours)
                │
         ┌──────┴─────────────────────────┐
         ▼                                ▼
[ Supervised Classifiers ]       [ Unsupervised Isolation Forest ]
(Random Forest / XGBoost)         (Behavioral Outlier Scoring)
         │                                │
         ▼                                ▼
   P(Fraud) [%]                    Anomaly Score [%]
         │                                │
         └──────────────┬─────────────────┘
                        ▼
       [ Composite Risk Assessment Layer ]
           (LOW / MEDIUM / HIGH)
                        │
                        ▼
       [ Domain Explainability Engine ]
                        │
                        ▼
       [ Interactive Streamlit Dashboard ]
```

---

## 4. DATA PREPROCESSING & FEATURE ENGINEERING

### 4.1 Schema Description
The dataset implements the 11 canonical PaySim fields:
- `step`: Simulation hour (1 to 744, covering 31 days).
- `type`: Transaction category (`PAYMENT`, `CASH_OUT`, `CASH_IN`, `TRANSFER`, `DEBIT`).
- `amount`: Monetary value in local currency units.
- `nameOrig`: Unique customer originator identifier (`C...`).
- `oldbalanceOrg`: Sender balance prior to transaction.
- `newbalanceOrig`: Sender balance subsequent to transaction.
- `nameDest`: Recipient customer (`C...`) or merchant (`M...`).
- `oldbalanceDest`: Recipient balance prior to transaction.
- `newbalanceDest`: Recipient balance subsequent to transaction.
- `isFraud`: Binary ground truth label (1 = Fraud, 0 = Legitimate).
- `isFlaggedFraud`: Regulatory threshold flag for transfers > $200,000.

### 4.2 Feature Engineering Formulations
To expose fraud dynamics, 23 domain features are derived using transaction amounts, balance differences, transaction types, and temporal patterns:

1. **Origin Balance Discrepancy Error:**
   $$\text{error\_balance\_orig} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$$
   *Rationale:* For authentic debits, the new balance equals old balance minus amount. A discrepancy exposes manipulated accounts.

2. **Destination Balance Discrepancy Error:**
   $$\text{error\_balance\_dest} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$$

3. **Account Draining Indicator:**
   $$\text{is\_drained\_orig} = \begin{cases} 1 & \text{if } \text{oldbalanceOrg} > 0 \text{ and } \text{newbalanceOrig} = 0 \\ 0 & \text{otherwise} \end{cases}$$

4. **Logarithmic Amount:**
   $$\text{log\_amount} = \ln(1 + \text{amount})$$

5. **Temporal Cyclical Hour:**
   $$\text{hour\_of\_day} = \text{step} \pmod{24}$$

---

## 5. MACHINE LEARNING & ANOMALY DETECTION ALGORITHMS

### 5.1 Supervised Models
1. **Logistic Regression (Baseline):** Linear classifier utilizing balanced class weights to penalize minority misclassifications.
2. **Random Forest Classifier (Selected):** Bagging ensemble of 100 decorrelated decision trees (`max_depth=12`, `class_weight='balanced'`). Gini impurity measures split quality.
3. **XGBoost Classifier:** Extreme Gradient Boosting optimizing regularized log-loss with scale positive weight $w = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 94.24$.

### 5.2 Unsupervised Isolation Forest
Isolation Forest isolates anomalies by randomly partitioning feature space with orthogonal hyperplanes:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
Where $h(x)$ represents the path length required to isolate observation $x$, and $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree (BST). Anomalous points exhibit systematically shorter path lengths.

---

## 6. EMPIRICAL RESULTS & BENCHMARKING

### 6.1 Performance Comparison Table (Holdout Test Set: 12,000 Samples, 126 Frauds)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Training Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 98.10% | 35.34% | **97.62%** | 51.90% | 0.9986 | 0.9542 | 0.13 s |
| **Random Forest (Winner)** | **99.87%** | 91.67% | 96.03% | **93.80%** | **0.9996** | **0.9814** | 0.57 s |
| **XGBoost Classifier** | **99.87%** | **92.97%** | 94.44% | 93.70% | 0.9993 | 0.9768 | 0.29 s |

### 6.2 Confusion Matrix (Random Forest Test Set)
- **True Negatives (Legitimate Correctly Cleared):** 11,863
- **False Positives (False Alarms):** 11 (Precision: 91.67%)
- **False Negatives (Missed Frauds):** 5 (Recall: 96.03%)
- **True Positives (Caught Frauds):** 121

### 6.3 Academic Insights
1. While Logistic Regression achieved high recall (97.62%), its precision was poor (35.34%), causing over 220 false alarms.
2. Random Forest and XGBoost achieved near-identical performance, with Random Forest achieving the highest F1-Score (93.80%) and highest PR-AUC (0.9814).
3. The dominant features driving decisions are `error_balance_orig`, `is_drained_orig`, and `amount_to_oldbalance_ratio`.

---

## 7. RISK STRATIFICATION & EXPLAINABILITY ENGINE

Predictions are categorized using a dual-engine decision matrix:
- **HIGH RISK:** $P(\text{Fraud}) \ge 70\%$ OR ($P(\text{Fraud}) \ge 40\%$ AND Isolation Forest flags anomaly).
- **MEDIUM RISK:** $25\% \le P(\text{Fraud}) < 70\%$ OR Anomaly detected.
- **LOW RISK:** $P(\text{Fraud}) < 25\%$ and normal behavioral profile.

Every output includes granular explainability tags (e.g., "Account balance completely depleted to $0.00", "Discrepancy of $X detected on origin balance").

---

## 8. TESTING & VALIDATION SUMMARY
The project incorporates an automated test suite (`tests/test_pipeline.py`) validating:
- Schema completeness across all 11 PaySim columns.
- Data cleaning sanitation (zero nulls, non-negative bounds).
- Feature matrix dimensionality (23 columns).
- Inference reliability on known normal transactions (returns LOW risk).
- Inference accuracy on account takeover scenarios (returns HIGH risk with drain indicators).
- Mathematical integrity of all evaluation metrics.

All 6 test cases passed successfully in 0.398s.

---

## 9. LIMITATIONS & ETHICAL CONSIDERATIONS
1. **Synthetic Nature of PaySim-Style Dataset:** Although statistically faithful to mobile money transaction mechanisms (Lopez-Rojas et al.), real-world banking features (IP geolocation, device finger-printing, biometrics) are absent.
2. **Compliance Support Role:** The system is engineered as an analytical decision-support tool for compliance officers, not an autonomous agent to freeze accounts without human oversight.

---

## 10. FUTURE SCOPE & CONCLUSION

### 10.1 Future Scope
- **Streaming Pipeline:** Ingestion via Apache Kafka with sub-second inference.
- **Graph Neural Networks (GNNs):** Modeling transaction networks using bipartite graphs to detect mule rings and syndicates.
- **SHAP / TreeExplainer:** Dynamic game-theoretic feature attribution.

### 10.2 Conclusion
The developed system demonstrates that combining domain-specific feature engineering with ensemble learning and unsupervised Isolation Forest effectively solves financial fraud detection under severe class imbalance. Random Forest achieved a 93.80% F1-score with 96.03% recall, and the Streamlit interface provides an intuitive platform for academic review and operational compliance.

---
**Report compiled for submission to Gujarat Technological University (GTU) Semester 7 examination.**
