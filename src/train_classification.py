import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger, save_model, save_json
from src.data_preprocessing import load_dataset, clean_data
from src.feature_engineering import engineer_features
from src.evaluation import evaluate_classifier, save_comparison_summary

def train_fraud_classifiers():
    """
    Orchestrates the training and evaluation of multiple supervised ML models:
    1. Logistic Regression (Linear baseline with class weights)
    2. Random Forest (Bagging ensemble)
    3. XGBoost (Gradient Boosting ensemble)
    
    Identifies the best model based on F1-Score & PR-AUC, and persists it.
    """
    logger.info("================ STARTING SUPERVISED CLASSIFICATION PIPELINE ================")
    start_time = time.time()
    
    # 1. Load and clean data
    raw_df = load_dataset()
    df, clean_report = clean_data(raw_df)
    
    # 2. Feature Engineering
    X, y, feature_names, scaler = engineer_features(df, is_training=True)
    
    # 3. Stratified Train/Test Split (80% Train, 20% Test)
    logger.info("Executing Stratified Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Training set: {len(X_train):,} rows ({y_train.sum()} fraud)")
    logger.info(f"Testing set:  {len(X_test):,} rows ({y_test.sum()} fraud)")
    
    # Calculate class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = float(neg_count / max(1, pos_count))
    logger.info(f"Class imbalance ratio: {neg_count}:{pos_count} (scale_pos_weight: {scale_weight:.2f})")
    
    # Scale continuous columns for Logistic Regression
    preproc_info = load_dataset_preproc = scaler
    continuous_cols = [
        'step', 'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'orig_balance_diff',
        'dest_balance_diff', 'error_balance_orig', 'error_balance_dest',
        'amount_to_oldbalance_ratio', 'hour_of_day', 'day_of_month'
    ]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # Define models dictionary
    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            "use_scaled": True
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            "use_scaled": False
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_weight,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False
        }
    }
    
    evaluation_results = []
    trained_models = {}
    
    for name, config in models.items():
        logger.info(f"\n--- Training {name} ---")
        model = config["estimator"]
        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_te = X_test_scaled if config["use_scaled"] else X_test
        
        m_start = time.time()
        model.fit(X_tr, y_train)
        m_time = time.time() - m_start
        logger.info(f"{name} trained in {m_time:.2f} seconds.")
        
        # Evaluate
        eval_metrics = evaluate_classifier(name, model, X_te, y_test, feature_names)
        eval_metrics["training_time_seconds"] = round(m_time, 2)
        eval_metrics["use_scaled"] = config["use_scaled"]
        evaluation_results.append(eval_metrics)
        trained_models[name] = model
        
    # Save comparison summary
    save_comparison_summary(evaluation_results)
    
    # Select best model based on F1-score & PR-AUC (fraud detection priority)
    best_eval = max(evaluation_results, key=lambda x: (x["f1_score"], x["pr_auc"]))
    best_model_name = best_eval["model_name"]
    best_model = trained_models[best_model_name]
    
    logger.info(f"\n🏆 Best Model Selected: {best_model_name}")
    logger.info(f"F1 Score: {best_eval['f1_score'] * 100:.2f}% | PR-AUC: {best_eval['pr_auc']:.4f} | Recall: {best_eval['recall'] * 100:.2f}%")
    
    # Persist best model
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "use_scaled": models[best_model_name]["use_scaled"],
        "threshold": 0.50,
        "metrics": best_eval
    }
    save_model(model_payload, "fraud_model.pkl")
    
    # Save metadata
    metadata = {
        "selected_model": best_model_name,
        "trained_at": datetime.now().isoformat(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "fraud_count_total": int(y.sum()),
        "features_count": len(feature_names),
        "feature_names": feature_names,
        "total_pipeline_time_seconds": round(time.time() - start_time, 2)
    }
    save_json(metadata, "model_metadata.json")
    
    logger.info("================ SUPERVISED CLASSIFICATION PIPELINE COMPLETE ================")
    return best_model_name, evaluation_results

if __name__ == "__main__":
    train_fraud_classifiers()
