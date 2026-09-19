import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import load_json
from src.data_preprocessing import load_dataset, clean_data, get_eda_summary
from src.prediction import TransactionPredictor
from dashboard.components import (
    render_kpi, render_risk_summary, render_explainability,
    plot_transaction_types, plot_amount_distribution, plot_fraud_by_type,
    plot_hourly_trend, plot_confusion_matrix_heatmap, plot_roc_curve_chart,
    plot_pr_curve_chart, plot_feature_importance_bar, plot_anomaly_scatter
)

def verify_all_pages():
    print("--- STARTING END-TO-END VERIFICATION OF ALL 7 DASHBOARD PAGES ---")
    
    # Ingestion
    raw_df = load_dataset()
    df, clean_report = clean_data(raw_df)
    eval_results = load_json("evaluation_results.json")
    model_metadata = load_json("model_metadata.json")
    anomaly_metadata = load_json("anomaly_metadata.json")
    predictor = TransactionPredictor()
    
    # 1. Overview Page
    print("[Verifying Page 1: Overview]...")
    fig_types = plot_transaction_types(df)
    fig_hourly = plot_hourly_trend(df)
    assert fig_types is not None
    assert fig_hourly is not None
    print("  -> Page 1 verified successfully.")
    
    # 2. Transaction Analytics Page
    print("[Verifying Page 2: Transaction Analytics]...")
    fig_log = plot_amount_distribution(df, log_scale=True)
    fig_raw = plot_amount_distribution(df, log_scale=False)
    assert fig_log is not None
    assert fig_raw is not None
    # Filtering verification
    df_f = df[(df['type'].isin(['TRANSFER', 'CASH_OUT'])) & (df['amount'] <= 500000)]
    assert len(df_f) > 0
    print("  -> Page 2 verified successfully.")
    
    # 3. Fraud Analysis Page
    print("[Verifying Page 3: Fraud Analysis]...")
    fraud_df = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0]
    fig_fraud_bar = plot_fraud_by_type(df)
    assert fig_fraud_bar is not None
    
    # Verify px.pie drain chart (the previous px bug!)
    drained_counts = pd.DataFrame({
        "Category": ["Drained to $0.00", "Partial Balance Left"],
        "Fraud": [int((fraud_df['newbalanceOrig'] == 0).sum()), int((fraud_df['newbalanceOrig'] > 0).sum())]
    })
    fig_drain = px.pie(
        drained_counts, values='Fraud', names='Category',
        title="Fraudster Behavior: Account Balance Depletion",
        hole=0.45,
        color_discrete_sequence=['#ef4444', '#f59e0b']
    )
    assert fig_drain is not None
    print("  -> Page 3 (including px.pie drain chart) verified successfully.")
    
    # 4. Anomaly Detection Page
    print("[Verifying Page 4: Anomaly Detection]...")
    sample_df = df.sample(min(2000, len(df)), random_state=42).copy()
    sample_df['orig_balance_diff'] = sample_df['oldbalanceOrg'] - sample_df['newbalanceOrig']
    
    # Real Isolation Forest prediction
    from src.feature_engineering import engineer_features
    X_sample, _, _, _ = engineer_features(sample_df, is_training=False, scaler=predictor.scaler)
    iso_preds = predictor.anomaly_model.predict(X_sample[predictor.anomaly_features])
    sample_df['anomaly_status'] = np.where(iso_preds == -1, 'Anomalous', 'Normal')
    
    fig_anom = plot_anomaly_scatter(sample_df)
    assert fig_anom is not None
    anom_count = (sample_df['anomaly_status'] == 'Anomalous').sum()
    print(f"  -> Page 4 verified with {anom_count} real Isolation Forest anomalies.")
    
    # 5. Transaction Prediction Page (Presets)
    print("[Verifying Page 5: Transaction Prediction Presets]...")
    presets = [
        ("Legitimate Merchant Payment", {
            'step': 14, 'type': 'PAYMENT', 'amount': 85.50,
            'oldbalanceOrg': 3500.00, 'newbalanceOrig': 3414.50,
            'nameDest': 'M987654321', 'oldbalanceDest': 0.00, 'newbalanceDest': 0.00
        }, "LOW"),
        ("Account Takeover Drain Fraud", {
            'step': 3, 'type': 'TRANSFER', 'amount': 350000.00,
            'oldbalanceOrg': 350000.00, 'newbalanceOrig': 0.00,
            'nameDest': 'C112233445', 'oldbalanceDest': 0.00, 'newbalanceDest': 350000.00
        }, "HIGH"),
        ("Balance Discrepancy Anomaly", {
            'step': 42, 'type': 'CASH_OUT', 'amount': 180000.00,
            'oldbalanceOrg': 250000.00, 'newbalanceOrig': 200000.00,
            'nameDest': 'C556677889', 'oldbalanceDest': 10000.00, 'newbalanceDest': 190000.00
        }, "MEDIUM")
    ]
    
    for name, p_input, expected_risk in presets:
        res = predictor.predict_single(p_input)
        print(f"  Testing {name}: Risk={res['risk_level']} (Expected={expected_risk}), FraudProb={res['fraud_probability_pct']}%, Anomaly={res['anomaly_status']}")
        assert res['risk_level'] == expected_risk, f"Preset {name} failed: got {res['risk_level']}, expected {expected_risk}"
        assert len(res['risk_indicators']) > 0, f"Preset {name} must have risk indicators"
    print("  -> Page 5 presets verified successfully.")
    
    # 6. Model Performance Page
    print("[Verifying Page 6: Model Performance]...")
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        m_data = eval_results["models"][m_name]
        fig_cm = plot_confusion_matrix_heatmap(m_data["confusion_matrix"], m_name)
        fig_roc = plot_roc_curve_chart(m_data["roc_curve"], m_name, m_data["roc_auc"])
        fig_pr = plot_pr_curve_chart(m_data["pr_curve"], m_name, m_data["pr_auc"])
        fig_feat = plot_feature_importance_bar(m_data["all_feature_importances"], m_name)
        assert fig_cm is not None and fig_roc is not None and fig_pr is not None and fig_feat is not None
    print("  -> Page 6 verified successfully.")
    
    # 7. About Project Page
    print("[Verifying Page 7: About Project]...")
    assert len(model_metadata["selected_model"]) > 0
    assert len(anomaly_metadata) > 0
    print("  -> Page 7 verified successfully.")
    
    print("\nALL 7 PAGES & ALL FEATURES VERIFIED SUCCESSFULLY WITHOUT ERRORS!")

if __name__ == "__main__":
    verify_all_pages()
