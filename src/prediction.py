import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.utils import logger, load_model
from src.feature_engineering import prepare_single_transaction, engineer_features

class TransactionPredictor:
    """
    Unified dual-engine inference system:
    1. Supervised Fraud Classification (Probability estimation)
    2. Unsupervised Anomaly Detection (Isolation Forest score)
    3. Tri-Tier Risk Level Classification (LOW, MEDIUM, HIGH)
    4. Academic Explainability Engine (Rule-based rationale + feature attribution)
    """
    
    def __init__(self):
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Loads all serialized models, preprocessors, and metadata."""
        logger.info("Loading prediction models and artifacts...")
        self.preproc = load_model("preprocessing.pkl")
        self.scaler = self.preproc["scaler"]
        self.continuous_cols = self.preproc["continuous"]
        
        self.fraud_payload = load_model("fraud_model.pkl")
        self.fraud_model = self.fraud_payload["model"]
        self.fraud_model_name = self.fraud_payload["model_name"]
        self.use_scaled = self.fraud_payload["use_scaled"]
        self.feature_names = self.fraud_payload["feature_names"]
        
        self.anomaly_payload = load_model("anomaly_model.pkl")
        self.anomaly_model = self.anomaly_payload["model"]
        self.anomaly_features = self.anomaly_payload["features"]
        self.score_min = self.anomaly_payload["score_min"]
        self.score_max = self.anomaly_payload["score_max"]
        logger.info("Artifacts successfully initialized.")

    def predict_single(self, tx_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction for an individual financial transaction.
        """
        # 1. Feature Engineering
        X_df = prepare_single_transaction(tx_dict, self.scaler)
        
        # 2. Supervised Fraud Probability
        X_input = X_df.copy()
        if self.use_scaled:
            X_input[self.continuous_cols] = self.scaler.transform(X_input[self.continuous_cols])
            
        if hasattr(self.fraud_model, "predict_proba"):
            fraud_prob = float(self.fraud_model.predict_proba(X_input)[0, 1])
        else:
            decision = self.fraud_model.decision_function(X_input)[0]
            fraud_prob = float(1.0 / (1.0 + np.exp(-decision)))
            
        fraud_pred = int(fraud_prob >= 0.50)
        
        # 3. Unsupervised Anomaly Detection
        X_anomaly = X_df[self.anomaly_features].copy()
        raw_anomaly_score = float(self.anomaly_model.decision_function(X_anomaly)[0])
        iso_pred = int(self.anomaly_model.predict(X_anomaly)[0])  # -1 = anomaly, 1 = normal
        is_anomaly = bool(iso_pred == -1)
        
        # Normalize anomaly score to [0, 1] range (higher = more anomalous)
        norm_anomaly_score = 1.0 - ((raw_anomaly_score - self.score_min) / (self.score_max - self.score_min + 1e-9))
        norm_anomaly_score = float(np.clip(norm_anomaly_score, 0.0, 1.0))
        
        # 4. Explainable Risk Indicators
        reasons = self._generate_explainability(tx_dict, X_df, fraud_prob, is_anomaly, norm_anomaly_score)
        
        # 5. Tri-Tier Risk Level Classification (integrates ML probability + anomaly + domain rules)
        risk_level, risk_color = self._determine_risk_level(fraud_prob, is_anomaly, norm_anomaly_score, reasons)
        
        return {
            "prediction": "Potentially Fraudulent" if fraud_pred == 1 else "Normal",
            "is_fraud_predicted": bool(fraud_pred == 1),
            "fraud_probability": round(fraud_prob, 4),
            "fraud_probability_pct": round(fraud_prob * 100, 2),
            "anomaly_status": "Anomalous" if is_anomaly else "Normal",
            "is_anomaly": is_anomaly,
            "anomaly_score": round(norm_anomaly_score, 4),
            "anomaly_score_pct": round(norm_anomaly_score * 100, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_indicators": reasons,
            "model_used": self.fraud_model_name,
            "engineered_features": X_df.iloc[0].to_dict()
        }

    def _determine_risk_level(
        self,
        fraud_prob: float,
        is_anomaly: bool,
        anomaly_score: float,
        indicators: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """
        Combines supervised and unsupervised signals with domain severity indicators:
        - HIGH: Fraud prob >= 0.70 OR (Fraud prob >= 0.35 AND is_anomaly) OR (has_critical AND fraud_prob >= 0.40)
        - MEDIUM: Fraud prob in [0.20, 0.70) OR is_anomaly OR anomaly_score >= 0.60 OR has_high OR has_critical
        - LOW: Otherwise
        """
        has_critical = any(ind.get("severity") == "CRITICAL" for ind in indicators)
        has_high = any(ind.get("severity") == "HIGH" for ind in indicators)

        if fraud_prob >= 0.70 or (fraud_prob >= 0.35 and is_anomaly) or (has_critical and fraud_prob >= 0.40):
            return "HIGH", "#dc3545"  # Red
        elif fraud_prob >= 0.20 or is_anomaly or anomaly_score >= 0.60 or has_high or has_critical:
            return "MEDIUM", "#fd7e14"  # Orange
        else:
            return "LOW", "#28a745"  # Green

    def _generate_explainability(
        self,
        raw_tx: Dict[str, Any],
        feat_df: pd.DataFrame,
        fraud_prob: float,
        is_anomaly: bool,
        anomaly_score: float
    ) -> List[Dict[str, str]]:
        """
        Generates human-readable, domain-backed explanations for the prediction.
        """
        indicators = []
        row = feat_df.iloc[0]
        ttype = str(raw_tx.get('type', '')).upper()
        amount = float(raw_tx.get('amount', 0.0))
        old_orig = float(raw_tx.get('oldbalanceOrg', 0.0))
        new_orig = float(raw_tx.get('newbalanceOrig', 0.0))
        old_dest = float(raw_tx.get('oldbalanceDest', 0.0))
        new_dest = float(raw_tx.get('newbalanceDest', 0.0))
        
        # 1. Total Account Drain Pattern
        if old_orig > 0 and new_orig == 0:
            indicators.append({
                "category": "Balance Draining",
                "severity": "CRITICAL",
                "message": f"Sender balance was completely depleted from ${old_orig:,.2f} to $0.00."
            })
            
        # 2. Origin Balance Discrepancy Error
        orig_err = abs(row['error_balance_orig'])
        if orig_err > 50.0 and ttype in ['TRANSFER', 'CASH_OUT']:
            indicators.append({
                "category": "Balance Discrepancy",
                "severity": "HIGH",
                "message": f"Discrepancy of ${orig_err:,.2f} detected between transaction amount and origin balance change."
            })
            
        # 3. Large Transaction Amount
        if amount >= 200000:
            indicators.append({
                "category": "High Value Outflow",
                "severity": "HIGH",
                "message": f"Transaction amount (${amount:,.2f}) surpasses the financial regulatory threshold of $200,000."
            })
        elif amount >= 100000:
            indicators.append({
                "category": "Elevated Volume",
                "severity": "MEDIUM",
                "message": f"Transaction amount (${amount:,.2f}) is in the upper 95th percentile of transaction amounts."
            })
            
        # 4. Zero Initial Balance Outflow
        if old_orig == 0 and amount > 0:
            indicators.append({
                "category": "Ghost Balance",
                "severity": "HIGH",
                "message": f"Transaction initiated from an account with zero balance (${amount:,.2f} transfer)."
            })
            
        # 5. Destination Account Profile
        if ttype == 'TRANSFER' and old_dest == 0:
            indicators.append({
                "category": "Unregistered Destination",
                "severity": "MEDIUM",
                "message": "Funds routed to a recipient account with zero historical balance."
            })
            
        # 6. Unsupervised Anomaly Signal
        if is_anomaly or anomaly_score > 0.65:
            indicators.append({
                "category": "Statistical Anomaly",
                "severity": "MEDIUM",
                "message": f"Isolation Forest identified out-of-distribution behavioral features (Score: {anomaly_score*100:.1f}%)."
            })
            
        # 7. Off-Peak Hour Activity
        hour = int(row['hour_of_day'])
        if 1 <= hour <= 5:
            indicators.append({
                "category": "Unusual Timing",
                "severity": "LOW",
                "message": f"Transaction processed during atypical off-peak overnight hours ({hour:02d}:00 hrs)."
            })
            
        # 8. High-Risk Channel
        if ttype in ['TRANSFER', 'CASH_OUT']:
            indicators.append({
                "category": "Vulnerable Channel",
                "severity": "INFO",
                "message": f"Transaction executed via {ttype}, the primary vector observed in mobile-money fraud."
            })
            
        # Default reassuring indicator if no suspicious patterns detected
        if not indicators:
            indicators.append({
                "category": "Routine Transaction",
                "severity": "INFO",
                "message": "Transaction conforms to expected baseline behavioral patterns and verified balances."
            })
            
        return indicators
