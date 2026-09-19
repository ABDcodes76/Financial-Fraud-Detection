import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from src.utils import logger, save_json

def evaluate_classifier(name: str, model, X_test, y_test, feature_names: List[str]) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics on test dataset:
    - Accuracy, Precision, Recall, F1 Score
    - ROC-AUC and PR-AUC (Average Precision)
    - Confusion Matrix (TP, FP, TN, FN)
    - ROC Curve points & PR Curve points
    - Top Feature Importances / Coefficients
    """
    logger.info(f"Evaluating model: {name}...")
    y_pred = model.predict(X_test)
    
    # Calculate probability estimates
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_proba = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)
        
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = 0.0
        
    cm = confusion_matrix(y_test, y_pred)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    # Compute downsampled ROC curve (max 100 points)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_step = max(1, len(fpr) // 100)
    roc_points = {
        "fpr": [round(float(x), 4) for x in fpr[::roc_step]],
        "tpr": [round(float(x), 4) for x in tpr[::roc_step]]
    }
    
    # Compute downsampled PR curve
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_step = max(1, len(p_curve) // 100)
    pr_points = {
        "precision": [round(float(x), 4) for x in p_curve[::pr_step]],
        "recall": [round(float(x), 4) for x in r_curve[::pr_step]]
    }
    
    # Extract feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for feat, imp in zip(feature_names, importances):
            feature_importance[feat] = round(float(imp), 5)
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        for feat, imp in zip(feature_names, coefs):
            feature_importance[feat] = round(float(imp), 5)
            
    # Sort top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    result = {
        "model_name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "matrix": [[tn, fp], [fn, tp]],
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "top_features": dict(sorted_features[:10]),
        "all_feature_importances": feature_importance
    }
    
    logger.info(
        f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
    )
    return result

def save_comparison_summary(results_list: List[Dict[str, Any]]):
    """Saves all model comparison metrics to JSON for dashboard visualization."""
    comparison = {
        "models": {res["model_name"]: res for res in results_list},
        "summary_table": [
            {
                "Model": res["model_name"],
                "Accuracy": f"{res['accuracy'] * 100:.2f}%",
                "Precision": f"{res['precision'] * 100:.2f}%",
                "Recall": f"{res['recall'] * 100:.2f}%",
                "F1 Score": f"{res['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{res['roc_auc']:.4f}",
                "PR-AUC": f"{res['pr_auc']:.4f}"
            }
            for res in results_list
        ]
    }
    save_json(comparison, "evaluation_results.json")
    return comparison
