import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger, save_model, save_json
from src.data_preprocessing import load_dataset, clean_data
from src.feature_engineering import engineer_features

ANOMALY_FEATURES = [
    'step',
    'amount',
    'log_amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'orig_balance_diff',
    'dest_balance_diff',
    'error_balance_orig',
    'error_balance_dest',
    'amount_to_oldbalance_ratio',
    'hour_of_day'
]

def train_isolation_forest(contamination: float = 0.03):
    """
    Trains an unsupervised Isolation Forest model to detect anomalous transactions.
    
    Distinction:
    - Fraud Classification: Supervised pattern matching against known fraud indicators.
    - Anomaly Detection: Unsupervised identification of rare, unusual, or out-of-distribution transactions.
    """
    logger.info("================ STARTING UNSUPERVISED ANOMALY DETECTION PIPELINE ================")
    start_time = time.time()
    
    # 1. Ingest clean data
    raw_df = load_dataset()
    df, _ = clean_data(raw_df)
    
    # 2. Extract domain features
    X, y, _, scaler = engineer_features(df, is_training=False)
    
    # Isolate unsupervised behavioral subset
    X_anomaly = X[ANOMALY_FEATURES].copy()
    
    logger.info(f"Training Isolation Forest with contamination={contamination} on {len(X_anomaly):,} samples...")
    iso_forest = IsolationForest(
        n_estimators=120,
        max_samples=256,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X_anomaly)
    
    # Anomaly scores: raw decision function (lower means more abnormal)
    raw_scores = iso_forest.decision_function(X_anomaly)
    preds = iso_forest.predict(X_anomaly)  # -1 = anomaly, 1 = normal
    
    # Normalize anomaly score to [0, 1] range where 1 is highest abnormality
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    norm_scores = 1.0 - ((raw_scores - score_min) / (score_max - score_min + 1e-9))
    
    is_anomaly = (preds == -1).astype(int)
    total_anomalies = int(is_anomaly.sum())
    anomaly_pct = float(is_anomaly.mean() * 100)
    
    # Calculate overlap with actual fraud label
    overlap_count = int(((is_anomaly == 1) & (y == 1)).sum())
    fraud_detected_pct = float((overlap_count / max(1, y.sum())) * 100)
    
    logger.info(f"Total Anomalies Flagged: {total_anomalies:,} ({anomaly_pct:.2f}%)")
    logger.info(f"Unsupervised Fraud Overlap: {overlap_count}/{y.sum()} frauds identified ({fraud_detected_pct:.2f}%)")
    
    # Persist anomaly model payload
    anomaly_payload = {
        "model": iso_forest,
        "features": ANOMALY_FEATURES,
        "contamination": contamination,
        "score_min": float(score_min),
        "score_max": float(score_max),
        "threshold": 0.50
    }
    save_model(anomaly_payload, "anomaly_model.pkl")
    
    # Save anomaly analytics summary
    anomaly_stats = {
        "total_transactions": len(df),
        "total_anomalies": total_anomalies,
        "anomaly_percentage": round(anomaly_pct, 2),
        "contamination": contamination,
        "fraud_overlap_count": overlap_count,
        "total_frauds": int(y.sum()),
        "fraud_overlap_rate": round(fraud_detected_pct, 2),
        "score_mean": round(float(norm_scores.mean()), 4),
        "score_std": round(float(norm_scores.std()), 4),
        "training_time_seconds": round(time.time() - start_time, 2)
    }
    save_json(anomaly_stats, "anomaly_metadata.json")
    
    logger.info("================ ANOMALY DETECTION PIPELINE COMPLETE ================")
    return anomaly_stats

if __name__ == "__main__":
    train_isolation_forest()
