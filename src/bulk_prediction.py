import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.utils import logger, load_model
from src.feature_engineering import engineer_features, FEATURE_NAMES

class BulkPredictionEngine:
    """
    High-throughput batch inference engine for financial fraud surveillance.
    Processes uploaded datasets in vectorized batches, computing:
    1. Supervised Fraud Probabilities & Classification
    2. Unsupervised Isolation Forest Anomaly Scores
    3. Multi-tier Risk Stratification (HIGH / MEDIUM / LOW)
    4. Primary Behavioral Flag Indicators
    5. Conditional Ground Truth Model Evaluation Metrics
    """

    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Initializing BulkPredictionEngine artifacts...")
        self.preproc = load_model("preprocessing.pkl")
        self.scaler = self.preproc["scaler"]
        self.continuous_cols = self.preproc["continuous"]

        self.fraud_payload = load_model("fraud_model.pkl")
        self.fraud_model = self.fraud_payload["model"]
        self.fraud_model_name = self.fraud_payload.get("model_name", "Random Forest Classifier")
        self.use_scaled = self.fraud_payload.get("use_scaled", False)
        self.feature_names = self.fraud_payload.get("feature_names", FEATURE_NAMES)

        self.anomaly_payload = load_model("anomaly_model.pkl")
        self.anomaly_model = self.anomaly_payload["model"]
        self.anomaly_features = self.anomaly_payload["features"]
        self.score_min = self.anomaly_payload.get("score_min", -0.2)
        self.score_max = self.anomaly_payload.get("score_max", 0.2)
        logger.info("BulkPredictionEngine ready for batch processing.")

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes end-to-end batch inference on an input DataFrame.
        Returns the original DataFrame enriched with all prediction and risk columns.
        """
        data = df.copy()
        n_rows = len(data)
        logger.info(f"Analyzing batch of {n_rows:,} transactions...")

        # 1. Vectorized Feature Engineering
        X_df, _, _, _ = engineer_features(data, is_training=False, scaler=self.scaler)

        # 2. Supervised Fraud Prediction
        X_input = X_df.copy()
        if self.use_scaled:
            X_input[self.continuous_cols] = self.scaler.transform(X_input[self.continuous_cols])

        if hasattr(self.fraud_model, "predict_proba"):
            fraud_probs = self.fraud_model.predict_proba(X_input)[:, 1]
        else:
            decision = self.fraud_model.decision_function(X_input)
            fraud_probs = 1.0 / (1.0 + np.exp(-decision))

        fraud_preds = (fraud_probs >= 0.50).astype(int)

        # 3. Unsupervised Anomaly Detection
        X_anomaly = X_df[self.anomaly_features].copy()
        raw_anomaly_scores = self.anomaly_model.decision_function(X_anomaly)
        iso_preds = self.anomaly_model.predict(X_anomaly)  # -1 = anomaly, 1 = normal
        is_anomalies = (iso_preds == -1)

        # Normalize anomaly scores to [0.0, 1.0] (higher = more anomalous)
        denom = (self.score_max - self.score_min) if (self.score_max - self.score_min) != 0 else 1.0
        norm_anomaly_scores = 1.0 - ((raw_anomaly_scores - self.score_min) / denom)
        norm_anomaly_scores = np.clip(norm_anomaly_scores, 0.0, 1.0)

        # 4. Compute Risk Levels and Primary Indicators
        risk_levels = []
        risk_scores = []
        primary_indicators = []

        types = data['type'].astype(str).str.upper().values
        amounts = data['amount'].values
        old_origs = data['oldbalanceOrg'].values
        new_origs = data['newbalanceOrig'].values
        old_dests = data['oldbalanceDest'].values
        new_dests = data['newbalanceDest'].values
        orig_errs = np.abs(X_df['error_balance_orig'].values)
        hours = X_df['hour_of_day'].values

        for i in range(n_rows):
            p = float(fraud_probs[i])
            is_anom = bool(is_anomalies[i])
            anom_s = float(norm_anomaly_scores[i])
            ttype = types[i]
            amt = float(amounts[i])
            old_o = float(old_origs[i])
            new_o = float(new_origs[i])
            old_d = float(old_dests[i])
            err_o = float(orig_errs[i])
            hr = int(hours[i])

            # Indicator detection
            reasons = []
            is_critical = False
            is_high_ind = False

            if old_o > 0 and new_o == 0:
                is_critical = True
                reasons.append("Complete Origin Balance Drain")

            if err_o > 50.0 and ttype in ['TRANSFER', 'CASH_OUT']:
                is_high_ind = True
                reasons.append(f"Balance Discrepancy (${err_o:,.0f})")

            if amt >= 200000:
                is_high_ind = True
                reasons.append(f"High Value (${amt:,.0f} >= $200k)")
            elif amt >= 100000:
                reasons.append("Elevated Value (>= $100k)")

            if old_o == 0 and amt > 0:
                is_high_ind = True
                reasons.append("Zero Initial Origin Balance")

            if ttype == 'TRANSFER' and old_d == 0:
                reasons.append("Unregistered Destination Account")

            if is_anom or anom_s > 0.65:
                reasons.append(f"Statistical Anomaly ({anom_s*100:.0f}%)")

            if 1 <= hr <= 5:
                reasons.append(f"Off-Peak Activity ({hr:02d}:00h)")

            if ttype in ['TRANSFER', 'CASH_OUT']:
                reasons.append(f"High-Risk Channel ({ttype})")

            # Tri-tier risk determination
            if p >= 0.70 or (p >= 0.35 and is_anom) or (is_critical and p >= 0.40):
                tier = "HIGH"
            elif p >= 0.20 or is_anom or anom_s >= 0.60 or is_high_ind or is_critical:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            risk_levels.append(tier)
            # Composite risk score (0-100)
            composite_score = round(max(p, anom_s * 0.7) * 100, 1)
            risk_scores.append(composite_score)
            
            primary_reason = " | ".join(reasons[:2]) if reasons else "Routine Baseline Activity"
            primary_indicators.append(primary_reason)

        # 5. Enrich DataFrame
        result_df = data.copy()
        result_df['fraud_probability'] = np.round(fraud_probs, 4)
        result_df['fraud_probability_pct'] = np.round(fraud_probs * 100, 2)
        result_df['fraud_prediction'] = np.where(fraud_preds == 1, 'Fraud', 'Normal')
        result_df['is_fraud_predicted'] = fraud_preds
        result_df['anomaly_score'] = np.round(norm_anomaly_scores, 4)
        result_df['anomaly_score_pct'] = np.round(norm_anomaly_scores * 100, 2)
        result_df['anomaly_status'] = np.where(is_anomalies, 'Anomalous', 'Normal')
        result_df['is_anomaly'] = is_anomalies.astype(int)
        result_df['risk_level'] = risk_levels
        result_df['risk_score'] = risk_scores
        result_df['primary_indicator'] = primary_indicators

        logger.info("Batch analysis complete.")
        return result_df

    def compute_summary_kpis(self, analyzed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes aggregate metrics, risk distribution, channel breakdown,
        and optional supervised evaluation metrics if 'isFraud' is available.
        """
        total_records = len(analyzed_df)
        if total_records == 0:
            return {}

        total_volume = float(analyzed_df['amount'].sum())
        predicted_fraud_count = int((analyzed_df['fraud_prediction'] == 'Fraud').sum())
        predicted_fraud_pct = (predicted_fraud_count / total_records) * 100

        anomalies_count = int((analyzed_df['anomaly_status'] == 'Anomalous').sum())
        anomalies_pct = (anomalies_count / total_records) * 100

        # Risk breakdown
        risk_counts = analyzed_df['risk_level'].value_counts().to_dict()
        high_risk_count = int(risk_counts.get('HIGH', 0))
        medium_risk_count = int(risk_counts.get('MEDIUM', 0))
        low_risk_count = int(risk_counts.get('LOW', 0))

        high_risk_volume = float(analyzed_df[analyzed_df['risk_level'] == 'HIGH']['amount'].sum())
        fraud_volume = float(analyzed_df[analyzed_df['fraud_prediction'] == 'Fraud']['amount'].sum())

        # Channel analysis
        channel_summary = {}
        for t in analyzed_df['type'].unique():
            sub = analyzed_df[analyzed_df['type'] == t]
            sub_fraud = int((sub['fraud_prediction'] == 'Fraud').sum())
            channel_summary[t] = {
                "count": len(sub),
                "volume": float(sub['amount'].sum()),
                "fraud_count": sub_fraud,
                "fraud_rate": round((sub_fraud / len(sub)) * 100, 2) if len(sub) > 0 else 0.0,
                "high_risk_count": int((sub['risk_level'] == 'HIGH').sum())
            }

        kpis = {
            "total_records": total_records,
            "total_volume": total_volume,
            "avg_transaction_amount": float(analyzed_df['amount'].mean()),
            "max_transaction_amount": float(analyzed_df['amount'].max()),
            "predicted_fraud_count": predicted_fraud_count,
            "predicted_fraud_pct": round(predicted_fraud_pct, 2),
            "predicted_fraud_volume": fraud_volume,
            "anomalies_count": anomalies_count,
            "anomalies_pct": round(anomalies_pct, 2),
            "high_risk_count": high_risk_count,
            "high_risk_pct": round((high_risk_count / total_records) * 100, 2),
            "high_risk_volume": high_risk_volume,
            "medium_risk_count": medium_risk_count,
            "medium_risk_pct": round((medium_risk_count / total_records) * 100, 2),
            "low_risk_count": low_risk_count,
            "low_risk_pct": round((low_risk_count / total_records) * 100, 2),
            "channel_summary": channel_summary,
            "model_name": self.fraud_model_name,
            "has_ground_truth": 'isFraud' in analyzed_df.columns
        }

        # Benchmark Ground Truth Evaluation
        if 'isFraud' in analyzed_df.columns:
            y_true = analyzed_df['isFraud'].astype(int).values
            y_pred = analyzed_df['is_fraud_predicted'].astype(int).values
            y_prob = analyzed_df['fraud_probability'].values

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)

            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
            try:
                if len(np.unique(y_true)) > 1:
                    roc_auc = float(roc_auc_score(y_true, y_prob))
                else:
                    roc_auc = 1.0
            except Exception:
                roc_auc = 0.0

            actual_fraud_count = int(np.sum(y_true))
            actual_fraud_pct = (actual_fraud_count / total_records) * 100

            kpis["evaluation"] = {
                "actual_fraud_count": actual_fraud_count,
                "actual_fraud_pct": round(actual_fraud_pct, 2),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn)
                }
            }

        return kpis
