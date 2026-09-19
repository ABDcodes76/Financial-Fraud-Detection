# GTU BE Computer Engineering Semester 7 — Live Project Demo Script
## Project: Financial Transaction Anomaly Detection & Fraud Risk Prediction System (FINSEC AI)

> **Estimated Time:** 5 to 7 Minutes  
> **Presenter:** GTU BE Computer Engineering Sem 7 Student  
> **Application URL:** `http://localhost:8501`  

---

## 🎬 Step-by-Step Viva Presentation Flow

### Step 1: Launch & Homepage Introduction (30 Seconds)
- **What to Click:** Open browser at `http://localhost:8501`, ensure Page **`01  Overview`** is selected.
- **What to Say:**
  > "Good morning respected examiners. I present **FINSEC AI**, an end-to-end Financial Transaction Risk Intelligence System developed for GTU Semester 7 in collaboration with InfoLabz IT Services Pvt. Ltd. Our platform combines supervised machine learning and unsupervised anomaly detection to solve the problem of financial fraud in mobile transactions."
- **What Faculty Should Notice:**
  - Modern, dark-mode FinTech UI with neon cyan branding and live telemetry status in the sidebar.
  - Top 4 KPI metric cards showing 60,000 Total Transactions, 630 Fraud Cases (1.05%), and $93.7M Total Fraud Incurred.
  - 24-Hour diurnal activity chart illustrating normal day-time peaks and night-time fraud penetration.

---

### Step 2: Transaction Analytics (45 Seconds)
- **What to Click:** In the sidebar radio button, select **`02  Transaction Analytics`**.
- **What to Say:**
  > "Here in Transaction Analytics, compliance officers can explore transaction volumes across 5 payment channels. As we can see, PAYMENT and CASH_OUT make up the largest volumes. The distribution histogram shows the heavily right-skewed transaction amounts, which we handle using log transformations during model feature engineering. We can dynamically filter transactions by channel and amount range."
- **What Faculty Should Notice:**
  - Fast, responsive data filtering without page lag.
  - Dual distribution views (Raw Amount vs Log-scale Amount).

---

### Step 3: Fraud Analysis (1 Minute)
- **What to Click:** Select **`03  Fraud Analysis`**.
- **What to Say:**
  > "Moving to Fraud Analysis, this reveals a critical empirical pattern in mobile financial networks: **100% of fraud occurs strictly within TRANSFER and CASH_OUT channels.** Fraudsters execute an unauthorized TRANSFER to compromise an account, followed immediately by a CASH_OUT to liquidate funds. Furthermore, over 80% of fraudulent transactions completely drain the sender's account to zero balance."
- **What Faculty Should Notice:**
  - Zero fraud in PAYMENT, CASH_IN, or DEBIT channels.
  - Zero-balance account draining donut chart highlighting that complete balance drainage is a primary indicator of account takeover.

---

### Step 4: Unsupervised Anomaly Detection (1 Minute)
- **What to Click:** Select **`04  Anomaly Detection`**.
- **What to Say:**
  > "While supervised models catch known fraud patterns, novel zero-day attacks require unsupervised learning. Here we implemented an **Isolation Forest** with a 3.0% contamination rate. This scatter plot runs real-time inference on 2,000 transactions, plotting Transaction Amount against Origin Balance Depletion. Anomalous transactions (marked in red) are isolated based on shorter partition path lengths in the isolation trees, capturing 35.5% of frauds without having seen any labels."
- **What Faculty Should Notice:**
  - Clear visual separation of normal clusters (blue) and anomalous points (red).
  - Accurate distinction between supervised fraud prediction and unsupervised anomaly detection.

---

### Step 5: Live Prediction Engine — 3 Presets (2.5 Minutes) ⭐ *CORE DEMO*
- **What to Click:** Select **`05  Transaction Prediction`**.

#### 5A. Preset 1 — Legitimate Merchant Payment
- **Action:** Click button **"Preset 1: Legitimate Merchant Payment"**, then click **"Run Risk Assessment"**.
- **What to Say:**
  > "Let us test Preset 1: a routine $450 payment to a merchant. The system evaluates both models: Supervised Fraud Probability is 0.0%, Isolation Forest marks it Normal. The assessed classification is **LOW RISK LEVEL** with green indicators."
- **What Faculty Should Notice:**
  - Green Low-Risk badge, 0.0% fraud probability bar, and 'CLEARED' indicators.

#### 5B. Preset 2 — Account Takeover / Account Drain
- **Action:** Click button **"Preset 2: Account Takeover Draining Fraud"**, then click **"Run Risk Assessment"**.
- **What to Say:**
  > "Now let us test Preset 2: a classic account takeover attack where $325,000 is transferred, draining the entire balance from $325,000 to $0.00. The system instantly detects this: Supervised Fraud Probability shoots to **96.99%**, triggering a **HIGH RISK LEVEL** alert. Notice the explainable diagnostics below: it automatically flags 100% account drainage and high-risk transfer channel."
- **What Faculty Should Notice:**
  - Red High-Risk badge, high fraud probability bar, and detailed explainability cards with `[CRITICAL]` badges.

#### 5C. Preset 3 — Balance Discrepancy Anomaly
- **Action:** Click button **"Preset 3: Balance Discrepancy Anomaly"**, then click **"Run Risk Assessment"**.
- **What to Say:**
  > "Preset 3 simulates a synthetic discrepancy attack: a $130,000 CASH_OUT where the sender account only had $15,000. While the raw model probability is moderate, our composite risk rules detect a severe $115,000 origin balance error and elevate the status to **MEDIUM RISK LEVEL** for investigator review."
- **What Faculty Should Notice:**
  - Amber Medium-Risk badge and clear discrepancy explanation showing intelligent decision support.

---

### Step 6: Model Performance Benchmarking (1 Minute)
- **What to Click:** Select **`06  Model Performance`**.
- **What to Say:**
  > "In Page 6, we benchmark our three supervised classifiers evaluated on a 20% holdout test set of 12,000 transactions. **Random Forest** achieved the best overall performance with **99.87% Accuracy, 91.67% Precision, 96.03% Recall, and 93.80% F1-Score**. Looking at the Confusion Matrix, it caught 121 out of 126 frauds with only 11 false alarms. The PR-AUC is 0.9814. Feature importance shows that our engineered balance difference error (`orig_error`) and balance drain ratio are the most predictive features."
- **What Faculty Should Notice:**
  - Real, un-fabricated metrics table matching the project report.
  - Interactive Confusion Matrix, ROC-AUC curve (0.9996), and PR-AUC curve (0.9814).

---

### Step 7: Conclusion & About Project (30 Seconds)
- **What to Click:** Select **`07  About Project`**.
- **What to Say:**
  > "Finally, Page 7 documents the GTU Semester 7 academic specifications, InfoLabz internship context, and the full pipeline architecture. All unit tests pass, the notebook is fully executed, and the system is production-ready. Thank you, I am now ready for questions."
- **What Faculty Should Notice:**
  - Complete academic compliance, clean Mermaid diagrams, and student-internship alignment.
