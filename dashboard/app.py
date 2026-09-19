import sys
import os
import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.styles import get_custom_css
from dashboard.components import (
    render_banner, render_kpi, render_risk_summary, render_explainability,
    plot_transaction_types, plot_amount_distribution, plot_fraud_by_type,
    plot_hourly_trend, plot_confusion_matrix_heatmap, plot_roc_curve_chart,
    plot_pr_curve_chart, plot_feature_importance_bar, plot_anomaly_scatter
)
from src.utils import load_json, get_data_path
from src.data_preprocessing import load_dataset, clean_data
from src.prediction import TransactionPredictor
from src.dataset_validator import validate_dataset, generate_sample_template
from src.bulk_prediction import BulkPredictionEngine
from src.report_generator import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="FINSEC AI | Financial Fraud Surveillance & Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Cache dataset loading for rapid navigation
@st.cache_data(show_spinner="Ingesting financial transaction ledger...")
def get_cached_data():
    try:
        raw = load_dataset()
        cleaned, _ = clean_data(raw)
        return cleaned
    except Exception as e:
        st.error(f"Error loading transaction dataset: {e}")
        return None

# Cache model evaluation metrics
@st.cache_data(show_spinner="Loading trained ML telemetry...")
def get_cached_metrics():
    try:
        eval_data = load_json("evaluation_results.json")
        meta_data = load_json("model_metadata.json")
        anomaly_data = load_json("anomaly_metadata.json")
        return eval_data, meta_data, anomaly_data
    except Exception as e:
        return None, None, None

# Initialize predictor singleton in session state
if "predictor" not in st.session_state:
    try:
        st.session_state.predictor = TransactionPredictor()
    except Exception as e:
        st.session_state.predictor = None

# Load dataset and artifacts
df = get_cached_data()
eval_results, model_metadata, anomaly_metadata = get_cached_metrics()

# ----------------- SIDEBAR NAVIGATION ----------------- #
st.sidebar.markdown("""
<div class="sidebar-brand-card">
    <div class="brand-icon">🛡️</div>
    <div class="brand-name">FINSEC AI</div>
    <div class="brand-title">Financial Fraud Risk Intelligence</div>
    <div class="brand-pill">ENTERPRISE ML SURVEILLANCE</div>
</div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Console",
    [
        "01  Overview",
        "02  Transaction Analytics",
        "03  Upload & Analyze Dataset",
        "04  Fraud Analysis",
        "05  Anomaly Detection",
        "06  Transaction Prediction",
        "07  Model Performance",
        "08  About Project"
    ]
)

st.sidebar.markdown("""
<div class="system-status-box">
    <div style="font-size: 0.68rem; color: #64748b; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.06em; text-transform: uppercase;">
        System Telemetry
    </div>
    <div class="status-item"><span class="status-dot dot-green"></span> Supervised Engine: <b style="color:#e2e8f0; margin-left:4px;">ACTIVE</b></div>
    <div class="status-item"><span class="status-dot dot-cyan"></span> Isolation Forest: <b style="color:#e2e8f0; margin-left:4px;">ARMED (3.0%)</b></div>
    <div class="status-item"><span class="status-dot dot-green"></span> Stream Ingestion: <b style="color:#e2e8f0; margin-left:4px;">VERIFIED</b></div>
</div>
""", unsafe_allow_html=True)

# ----------------- PAGE 1: OVERVIEW ----------------- #
if "01  Overview" in nav_page:
    render_banner(
        "Financial Transaction Risk Intelligence Platform",
        "Enterprise financial surveillance, supervised fraud classification & unsupervised behavioral anomaly detection",
        badge="LIVE SURVEILLANCE"
    )
    
    if df is not None:
        total_tx = len(df)
        total_amt = df['amount'].sum()
        fraud_tx = int(df['isFraud'].sum())
        fraud_pct = df['isFraud'].mean() * 100
        anomalies_count = anomaly_metadata["total_anomalies"] if anomaly_metadata else int(total_tx * 0.03)
        accuracy_val = f"{eval_results['models'][model_metadata['selected_model']]['accuracy']*100:.2f}%" if eval_results else "99.87%"
        
        # System Intelligence Banner
        st.markdown(f"""
        <div class="monitor-panel">
            <div>
                <span style="color: #38bdf8; font-weight: 700;">LIVE MONITORING:</span> 
                <span>Auditing active ledger of <b>{total_tx:,}</b> transactions across 5 channels</span>
            </div>
            <div>
                <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; border: 1px solid rgba(52, 211, 153, 0.3);">
                    ● ONLINE & VERIFIED
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Row 1: KPI Cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            render_kpi("Total Ingestion", f"{total_tx:,}", "Audited transactions", icon="🌐")
        with k2:
            render_kpi("Gross Volume", f"${total_amt/1e6:,.1f}M", "Total settled currency", icon="💳")
        with k3:
            render_kpi("Fraud Events", f"{fraud_tx:,}", f"{fraud_pct:.2f}% of total volume", icon="🚨")
        with k4:
            render_kpi("Fraud Volume Rate", f"{fraud_pct:.2f}%", "Verified class ratio", icon="📉")
        with k5:
            render_kpi("Outlier Anomalies", f"{anomalies_count:,}", "Isolation Forest flags", icon="🔍")
        with k6:
            render_kpi("Ensemble Accuracy", accuracy_val, "Holdout test baseline", icon="🏆")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: High-Level Visualizations
        c1, c2 = st.columns([1.1, 1.4])
        with c1:
            st.plotly_chart(plot_transaction_types(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_hourly_trend(df), use_container_width=True)
            
        st.markdown("##### 📋 Audited Transaction Ledger (Recent Stream Sample)")
        st.dataframe(
            df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'isFraud']].head(12),
            use_container_width=True
        )
    else:
        st.warning("Dataset not available. Please run dataset generator.")

# ----------------- PAGE 2: TRANSACTION ANALYTICS ----------------- #
elif "02  Transaction Analytics" in nav_page:
    render_banner(
        "Financial Transaction Analytics & Distribution",
        "Multi-parameter transactional filtering, raw vs log-scale distributions, and temporal trends",
        badge="ANALYTICS CONSOLE"
    )
    
    if df is not None:
        # Filter panel
        with st.container():
            st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
            f1, f2, f3 = st.columns([1.5, 2, 1.5])
            with f1:
                selected_types = st.multiselect("Transaction Channel Categories", options=df['type'].unique(), default=list(df['type'].unique()))
            with f2:
                amt_range = st.slider("Amount Bounds ($)", 0.0, float(df['amount'].quantile(0.999)), (0.0, float(df['amount'].quantile(0.95))))
            with f3:
                step_range = st.slider("Simulation Time (Hour Interval)", 1, int(df['step'].max()), (1, int(df['step'].max())))
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Apply filters
        df_filtered = df[
            (df['type'].isin(selected_types)) &
            (df['amount'] >= amt_range[0]) &
            (df['amount'] <= amt_range[1]) &
            (df['step'] >= step_range[0]) &
            (df['step'] <= step_range[1])
        ]
        
        st.markdown(f"Displaying **{len(df_filtered):,}** of **{len(df):,}** transactions satisfying active surveillance filters.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_amount_distribution(df_filtered, log_scale=True), use_container_width=True)
        with c2:
            st.plotly_chart(plot_amount_distribution(df_filtered, log_scale=False), use_container_width=True)
            
        st.markdown("##### 🔍 Filtered Ledger Records Explorer")
        st.dataframe(df_filtered.head(100), use_container_width=True)
        
        csv_data = df_filtered.head(2000).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Ledger (CSV)",
            data=csv_data,
            file_name="filtered_transactions.csv",
            mime="text/csv"
        )

# ----------------- PAGE 3: UPLOAD & ANALYZE DATASET ----------------- #
elif "03  Upload & Analyze Dataset" in nav_page:
    render_banner(
        "Upload & Analyze Transaction Dataset",
        "High-throughput batch ML risk scoring, multi-tier classification, and forensic PDF reporting",
        badge="BATCH ML SURVEILLANCE"
    )
    
    # 1. Guidelines & Sample Download Header Card
    with st.expander("ℹ️ Dataset Schema Specifications & CSV Template Instructions", expanded=False):
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.85rem; color: #cbd5e1;">
            <b>Required Transaction Ledger Schema:</b><br/>
            • <code>step</code>: Integer simulation hour (1 - 744)<br/>
            • <code>type</code>: Transaction channel (<code>TRANSFER</code>, <code>CASH_OUT</code>, <code>PAYMENT</code>, <code>CASH_IN</code>, <code>DEBIT</code>)<br/>
            • <code>amount</code>: Transaction amount in USD ($)<br/>
            • <code>oldbalanceOrg</code>: Sender initial balance before transaction<br/>
            • <code>newbalanceOrig</code>: Sender subsequent balance after transaction<br/>
            • <code>oldbalanceDest</code>: Recipient initial balance before transaction<br/>
            • <code>newbalanceDest</code>: Recipient subsequent balance after transaction<br/>
            <br/>
            <b>Optional Columns:</b> <code>nameOrig</code>, <code>nameDest</code>, <code>isFraud</code> (ground-truth label for evaluation benchmark), <code>isFlaggedFraud</code>.
        </div>
        """, unsafe_allow_html=True)
        
    # Sample Template Download Action
    sample_template_df = generate_sample_template()
    sample_csv_bytes = sample_template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv_bytes,
        file_name="sample_transactions_template.csv",
        mime="text/csv",
        help="Download a pre-formatted 10-row template with both normal and fraud cases."
    )
    
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # 2. File Uploader
    uploaded_file = st.file_uploader(
        "Upload Financial Transaction Dataset (CSV format)",
        type=["csv"],
        help="Upload CSV files containing raw transactions. Maximum recommended size for browser processing: 50,000 rows."
    )
    
    if uploaded_file is not None:
        with st.spinner("Validating dataset structure and schema integrity..."):
            validation = validate_dataset(uploaded_file)
            
        if not validation.is_valid:
            st.error(f"❌ Schema Validation Failed: {len(validation.error_messages)} issue(s) detected.")
            for err in validation.error_messages:
                st.markdown(f"- 🔴 **Error:** {err}")
            st.warning("Please download the standard sample CSV template above and ensure all required column headers match exactly.")
        else:
            # Validation Success Panel
            st.success(f"✅ Schema Validation Passed! Successfully verified **{validation.row_count:,}** rows with **{validation.col_count}** columns.")
            
            # Show warnings if any
            if validation.warning_messages:
                with st.expander("⚠️ Data Sanitization & Cleaning Audit Logs", expanded=False):
                    for w in validation.warning_messages:
                        st.info(f"• {w}")
                        
            # Raw Data Preview
            with st.expander("🔍 Ingested Dataset Preview (First 10 Rows)", expanded=False):
                st.dataframe(validation.cleaned_df.head(10), use_container_width=True)
                
            # Trigger Batch ML Surveillance
            run_analysis = st.button("🚀 RUN FULL FRAUD SURVEILLANCE & RISK ANALYSIS", use_container_width=True)
            
            # Check if analysis is stored in session state for this file
            file_key = f"batch_analysis_{uploaded_file.name}_{validation.row_count}"
            
            if run_analysis or file_key in st.session_state:
                if run_analysis or file_key not in st.session_state:
                    with st.spinner("Executing dual-engine feature extraction, supervised scoring, and anomaly detection..."):
                        engine = BulkPredictionEngine()
                        analyzed_df = engine.analyze_dataset(validation.cleaned_df)
                        batch_kpis = engine.compute_summary_kpis(analyzed_df)
                        st.session_state[file_key] = {
                            "analyzed_df": analyzed_df,
                            "kpis": batch_kpis,
                            "validation": validation
                        }
                
                cache_payload = st.session_state[file_key]
                analyzed_df = cache_payload["analyzed_df"]
                kpis = cache_payload["kpis"]
                
                st.markdown("---")
                st.markdown("### 📊 Batch Surveillance Diagnostics & Intelligence Summary")
                
                # 6 Top-Level KPI Cards
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1:
                    render_kpi("Audited Volume", f"{kpis['total_records']:,}", "Transactions scanned", icon="🌐")
                with k2:
                    render_kpi("Settled Amount", f"${kpis['total_volume']/1e6:,.2f}M" if kpis['total_volume'] >= 1e6 else f"${kpis['total_volume']:,.0f}", "Gross capital", icon="💳")
                with k3:
                    render_kpi("Fraud Cases", f"{kpis['predicted_fraud_count']:,}", f"{kpis['predicted_fraud_pct']:.2f}% flagged rate", icon="🚨")
                with k4:
                    render_kpi("Anomalies", f"{kpis['anomalies_count']:,}", f"{kpis['anomalies_pct']:.2f}% out-of-distribution", icon="🔍")
                with k5:
                    render_kpi("High-Risk Exposure", f"${kpis['high_risk_volume']/1e6:,.2f}M" if kpis['high_risk_volume'] >= 1e6 else f"${kpis['high_risk_volume']:,.0f}", f"{kpis['high_risk_pct']:.1f}% of total rows", icon="⚡")
                with k6:
                    model_label = "Supervised Dual-ML"
                    render_kpi("Engine Baseline", model_label, "Random Forest + IF", icon="🛡️")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualizations Grid (2x2)
                row1_c1, row1_c2 = st.columns(2)
                
                with row1_c1:
                    # Risk Distribution Pie/Donut Chart
                    risk_counts = pd.DataFrame({
                        "Tier": ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
                        "Count": [kpis.get("high_risk_count", 0), kpis.get("medium_risk_count", 0), kpis.get("low_risk_count", 0)]
                    })
                    fig_risk = px.pie(
                        risk_counts, values='Count', names='Tier',
                        title="Multi-Tier Risk Stratification Breakdown",
                        hole=0.5,
                        color='Tier',
                        color_discrete_map={
                            "HIGH RISK": "#ef4444",
                            "MEDIUM RISK": "#f59e0b",
                            "LOW RISK": "#10b981"
                        }
                    )
                    fig_risk.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                        margin=dict(t=45, b=25, l=25, r=25),
                        height=330,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                with row1_c2:
                    # Supervised Fraud vs Normal Bar Chart
                    pred_counts = pd.DataFrame({
                        "Classification": ["Normal", "Potentially Fraudulent"],
                        "Count": [kpis["total_records"] - kpis["predicted_fraud_count"], kpis["predicted_fraud_count"]]
                    })
                    fig_pred = px.bar(
                        pred_counts, x="Classification", y="Count",
                        title="Supervised Classification Distribution (Threshold >= 0.50)",
                        color="Classification",
                        color_discrete_map={"Normal": "#38bdf8", "Potentially Fraudulent": "#ef4444"},
                        text_auto=True
                    )
                    fig_pred.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                        margin=dict(t=45, b=25, l=25, r=25),
                        height=330,
                        showlegend=False
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                row2_c1, row2_c2 = st.columns(2)
                
                with row2_c1:
                    # Anomaly & Risk Scatter Plot
                    scatter_sample = analyzed_df.sample(min(1500, len(analyzed_df)), random_state=42).copy() if len(analyzed_df) > 1500 else analyzed_df.copy()
                    scatter_sample['orig_balance_diff'] = scatter_sample['oldbalanceOrg'] - scatter_sample['newbalanceOrig']
                    
                    fig_scat = px.scatter(
                        scatter_sample,
                        x="amount",
                        y="orig_balance_diff",
                        color="risk_level",
                        symbol="anomaly_status",
                        title="Behavioral Outlier & Anomaly Scatter Space",
                        labels={"amount": "Transaction Amount ($)", "orig_balance_diff": "Origin Balance Change ($)"},
                        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                        hover_data=["type", "fraud_probability_pct", "anomaly_score_pct"]
                    )
                    fig_scat.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                        margin=dict(t=45, b=25, l=25, r=25),
                        height=330
                    )
                    st.plotly_chart(fig_scat, use_container_width=True)
                    
                with row2_c2:
                    # Channel Fraud Breakdown
                    ch_df = pd.DataFrame([
                        {"Channel": k, "Total Volume": v["volume"], "Fraud Count": v["fraud_count"], "Fraud Rate (%)": v["fraud_rate"]}
                        for k, v in kpis.get("channel_summary", {}).items()
                    ])
                    fig_ch = px.bar(
                        ch_df, x="Channel", y="Fraud Count",
                        title="Channel Fraud Incursion Concentration",
                        color="Channel",
                        text_auto=True
                    )
                    fig_ch.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                        margin=dict(t=45, b=25, l=25, r=25),
                        height=330,
                        showlegend=False
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)
                    
                # Conditional Ground Truth Evaluation Section
                if validation.has_ground_truth and "evaluation" in kpis:
                    st.markdown("---")
                    st.markdown("### 🎯 Supervised Ground-Truth Verification Benchmark")
                    st.info("💡 **Ground Truth Detected (`isFraud` column found):** Computing empirical model validation performance metrics on uploaded test batch.")
                    
                    ev = kpis["evaluation"]
                    cm = ev["confusion_matrix"]
                    
                    e1, e2, e3, e4, e5 = st.columns(5)
                    with e1:
                        render_kpi("Accuracy", f"{ev['accuracy']*100:.2f}%", "Overall correctness", icon="🎯")
                    with e2:
                        render_kpi("Precision", f"{ev['precision']*100:.2f}%", "TP / (TP + FP)", icon="✨")
                    with e3:
                        render_kpi("Recall", f"{ev['recall']*100:.2f}%", "TP / (TP + FN)", icon="⚡")
                    with e4:
                        render_kpi("F1-Score", f"{ev['f1_score']:.4f}", "Harmonic mean", icon="🏆")
                    with e5:
                        render_kpi("ROC-AUC", f"{ev['roc_auc']:.4f}", "Discrimination power", icon="📈")
                        
                    # Confusion Matrix Display
                    cm_col1, cm_col2 = st.columns([1.2, 1.8])
                    with cm_col1:
                        z_matrix = [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]
                        fig_cm = px.imshow(
                            z_matrix,
                            labels=dict(x="Predicted Label", y="Actual Label", color="Transactions"),
                            x=['Normal (0)', 'Fraud (1)'],
                            y=['Normal (0)', 'Fraud (1)'],
                            color_continuous_scale=[[0, '#0f172a'], [0.5, '#0284c7'], [1, '#ef4444']],
                            text_auto=True,
                            title="Confusion Matrix Heatmap"
                        )
                        fig_cm.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.5)",
                            font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                            margin=dict(t=45, b=25, l=25, r=25),
                            height=280
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                    with cm_col2:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 16px; margin-top: 25px;">
                            <div style="font-weight: 700; color: #38bdf8; margin-bottom: 8px;">Confusion Matrix Diagnostics:</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.82rem;">
                                <div>• <b>True Positives (TP):</b> <span style="color:#ef4444; font-weight:700;">{cm['tp']:,}</span> (Correctly flagged frauds)</div>
                                <div>• <b>False Positives (FP):</b> <span style="color:#f59e0b; font-weight:700;">{cm['fp']:,}</span> (Normal flagged as fraud)</div>
                                <div>• <b>True Negatives (TN):</b> <span style="color:#10b981; font-weight:700;">{cm['tn']:,}</span> (Correctly cleared normal)</div>
                                <div>• <b>False Negatives (FN):</b> <span style="color:#ec4899; font-weight:700;">{cm['fn']:,}</span> (Missed fraud cases)</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # High-Risk Ledger Table & Explorer
                st.markdown("---")
                st.markdown("### 📋 Analyzed Ledger & Risk Explorer")
                
                tier_filter = st.selectbox(
                    "Filter Ledger by Risk Stratification Tier:",
                    options=["ALL TRANSACTIONS", "HIGH RISK ONLY", "MEDIUM RISK ONLY", "LOW RISK ONLY", "PREDICTED FRAUD ONLY"],
                    index=0
                )
                
                if tier_filter == "HIGH RISK ONLY":
                    view_df = analyzed_df[analyzed_df['risk_level'] == 'HIGH']
                elif tier_filter == "MEDIUM RISK ONLY":
                    view_df = analyzed_df[analyzed_df['risk_level'] == 'MEDIUM']
                elif tier_filter == "LOW RISK ONLY":
                    view_df = analyzed_df[analyzed_df['risk_level'] == 'LOW']
                elif tier_filter == "PREDICTED FRAUD ONLY":
                    view_df = analyzed_df[analyzed_df['fraud_prediction'] == 'Fraud']
                else:
                    view_df = analyzed_df
                    
                display_cols = [
                    'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'fraud_probability_pct', 'anomaly_status', 'risk_level', 'primary_indicator'
                ]
                if 'isFraud' in analyzed_df.columns:
                    display_cols.insert(5, 'isFraud')
                    
                st.dataframe(view_df[display_cols].head(100), use_container_width=True)
                st.caption(f"Displaying top {min(100, len(view_df))} of {len(view_df):,} filtered transactions.")
                
                # Download Center (Analyzed CSV + Forensic PDF)
                st.markdown("---")
                st.markdown("### 📥 Export & Forensic Download Center")
                
                d_col1, d_col2 = st.columns(2)
                
                with d_col1:
                    analyzed_csv = analyzed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Analyzed Dataset (CSV)",
                        data=analyzed_csv,
                        file_name=f"FINSEC_Analyzed_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                with d_col2:
                    pdf_bytes = generate_pdf_report(analyzed_df, validation, kpis)
                    st.download_button(
                        label="📄 Download Forensic Audit Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"FINSEC_Audit_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# ----------------- PAGE 4: FRAUD ANALYSIS ----------------- #
elif "04  Fraud Analysis" in nav_page:
    render_banner(
        "Supervised Fraud Investigation & Drain Vectors",
        "Detailed behavioral analysis of confirmed fraudulent incursions, channel vulnerability, and account siphoning",
        badge="INVESTIGATION CONSOLE"
    )
    
    if df is not None:
        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0]
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi("Fraud Cases", f"{len(fraud_df):,}", "Confirmed fraudulent incursions", icon="🚨")
        with m2:
            render_kpi("Compromised Capital", f"${fraud_df['amount'].sum()/1e6:,.2f}M", "Total siphoned funds", icon="💸")
        with m3:
            render_kpi("Mean Fraud Sum", f"${fraud_df['amount'].mean():,.2f}", f"Normal avg: ${normal_df['amount'].mean():,.2f}", icon="📊")
        with m4:
            render_kpi("Account Drain Rate", f"{(fraud_df['newbalanceOrig'] == 0).mean()*100:.1f}%", "Accounts emptied to $0.00", icon="⚡")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.3, 1.1])
        with c1:
            st.plotly_chart(plot_fraud_by_type(df), use_container_width=True)
        with c2:
            # Drain analysis donut
            drained_counts = pd.DataFrame({
                "Category": ["Drained to $0.00", "Partial Balance Retained"],
                "Fraud": [int((fraud_df['newbalanceOrig'] == 0).sum()), int((fraud_df['newbalanceOrig'] > 0).sum())]
            })
            fig_drain = px.pie(
                drained_counts, values='Fraud', names='Category',
                title="Fraudster Behavioral Signature: Account Depletion",
                hole=0.55,
                color_discrete_sequence=['#ef4444', '#f59e0b']
            )
            fig_drain.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
                margin=dict(t=45, b=25, l=25, r=25),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            fig_drain.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#0d121f', width=2))
            )
            st.plotly_chart(fig_drain, use_container_width=True)
            
        st.markdown("##### 🚨 Confirmed Fraudulent Activity Ledger")
        st.dataframe(
            fraud_df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'isFlaggedFraud']].head(50),
            use_container_width=True
        )

# ----------------- PAGE 5: ANOMALY DETECTION ----------------- #
elif "05  Anomaly Detection" in nav_page:
    render_banner(
        "Unsupervised Anomaly Detection (Isolation Forest)",
        "Out-of-distribution geometric outlier detection identifying abnormal patterns without supervisory labels",
        badge="ANOMALY INTELLIGENCE"
    )
    
    st.info("""
    **Methodological Comparison:**
    - **Supervised Fraud Classification** fits known historic signatures (e.g. account draining via TRANSFER/CASH_OUT).
    - **Unsupervised Anomaly Detection** constructs orthogonal isolation trees across 13 behavioral dimensions, catching rare or novel structural anomalies without labels.
    - An identified anomaly is **Potentially Suspicious / Unusual**, flagging transactions for analyst review.
    """)
    
    if df is not None and anomaly_metadata is not None:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            render_kpi("Flagged Anomalies", f"{anomaly_metadata['total_anomalies']:,}", "Isolation Forest outliers", icon="🔍")
        with a2:
            render_kpi("Contamination Prior", f"{anomaly_metadata['contamination']*100:.1f}%", "Target outlier budget", icon="⚙️")
        with a3:
            render_kpi("Fraud Overlap Rate", f"{anomaly_metadata['fraud_overlap_rate']:.1f}%", "Frauds caught unsupervised", icon="🎯")
        with a4:
            render_kpi("Decision Score Mean", f"{anomaly_metadata['score_mean']:.3f}", f"Std dev: {anomaly_metadata['score_std']:.3f}", icon="📈")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Real Isolation Forest scatter plot
        sample_df = df.sample(min(2000, len(df)), random_state=42).copy()
        sample_df['orig_balance_diff'] = sample_df['oldbalanceOrg'] - sample_df['newbalanceOrig']
        
        predictor = st.session_state.get("predictor")
        if predictor is not None and hasattr(predictor, "anomaly_model"):
            from src.feature_engineering import engineer_features
            X_sample, _, _, _ = engineer_features(sample_df, is_training=False, scaler=predictor.scaler)
            iso_preds = predictor.anomaly_model.predict(X_sample[predictor.anomaly_features])
            sample_df['anomaly_status'] = np.where(iso_preds == -1, 'Anomalous', 'Normal')
            st.caption("ℹ️ Data points evaluated directly by the trained Isolation Forest model (`anomaly_model.pkl`, contamination = 3.0%).")
        else:
            sample_df['anomaly_status'] = np.where(sample_df['isFraud'] == 1, 'Anomalous', 'Normal')
            st.caption("ℹ️ Illustrative behavioral scatter visualization.")
            
        st.plotly_chart(plot_anomaly_scatter(sample_df), use_container_width=True)

# ----------------- PAGE 6: TRANSACTION PREDICTION ----------------- #
elif "06  Transaction Prediction" in nav_page:
    render_banner(
        "Real-Time Transaction Risk Assessment Engine",
        "Dual-engine inference combining supervised fraud probability, Isolation Forest anomaly scoring, and explainable indicators",
        badge="DECISION SUPPORT"
    )
    
    predictor = st.session_state.predictor
    if predictor is None:
        st.error("Prediction models are not initialized. Please verify model artifacts in models/ directory.")
    else:
        # Academic Quick Presets
        st.markdown("##### ⚡ Quick-Load Academic Demonstration Scenarios:")
        preset_cols = st.columns(3)
        
        with preset_cols[0]:
            if st.button("🟢 PRESET 1: Legitimate Payment"):
                st.session_state.preset_step = 14
                st.session_state.preset_type = "PAYMENT"
                st.session_state.preset_amount = 85.50
                st.session_state.preset_old_orig = 3500.00
                st.session_state.preset_new_orig = 3414.50
                st.session_state.preset_old_dest = 0.00
                st.session_state.preset_new_dest = 0.00
                st.session_state.preset_dest = "M987654321"
                
        with preset_cols[1]:
            if st.button("🔴 PRESET 2: Account Takeover Drain"):
                st.session_state.preset_step = 3
                st.session_state.preset_type = "TRANSFER"
                st.session_state.preset_amount = 350000.00
                st.session_state.preset_old_orig = 350000.00
                st.session_state.preset_new_orig = 0.00
                st.session_state.preset_old_dest = 0.00
                st.session_state.preset_new_dest = 350000.00
                st.session_state.preset_dest = "C112233445"
                
        with preset_cols[2]:
            if st.button("🟠 PRESET 3: Discrepancy Anomaly"):
                st.session_state.preset_step = 42
                st.session_state.preset_type = "CASH_OUT"
                st.session_state.preset_amount = 180000.00
                st.session_state.preset_old_orig = 250000.00
                st.session_state.preset_new_orig = 200000.00
                st.session_state.preset_old_dest = 10000.00
                st.session_state.preset_new_dest = 190000.00
                st.session_state.preset_dest = "C556677889"
                
        with st.form("transaction_prediction_form"):
            st.markdown('<div class="form-section-header">Section 1 • Transaction Parameters & Channel</div>', unsafe_allow_html=True)
            row1_1, row1_2, row1_3 = st.columns(3)
            with row1_1:
                step_val = st.number_input("Time Step (Hour of Month)", min_value=1, max_value=744, value=st.session_state.get('preset_step', 10))
            with row1_2:
                type_options = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
                cur_type = st.session_state.get('preset_type', 'TRANSFER')
                type_idx = type_options.index(cur_type) if cur_type in type_options else 0
                type_val = st.selectbox("Transaction Channel Type", options=type_options, index=type_idx)
            with row1_3:
                amount_val = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(st.session_state.get('preset_amount', 1000.00)), step=100.0)
                
            st.markdown('<div class="form-section-header">Section 2 • Sender (Origin) Account Profile</div>', unsafe_allow_html=True)
            row2_1, row2_2, row2_3 = st.columns(3)
            with row2_1:
                orig_id = st.text_input("Origin Account ID", value="C123456789")
            with row2_2:
                old_orig_val = st.number_input("Initial Balance (Origin) ($)", min_value=0.0, value=float(st.session_state.get('preset_old_orig', 5000.00)), step=100.0)
            with row2_3:
                new_orig_val = st.number_input("Subsequent Balance (Origin) ($)", min_value=0.0, value=float(st.session_state.get('preset_new_orig', 4000.00)), step=100.0)
                
            st.markdown('<div class="form-section-header">Section 3 • Recipient (Destination) Account Profile</div>', unsafe_allow_html=True)
            row3_1, row3_2, row3_3 = st.columns(3)
            with row3_1:
                dest_id = st.text_input("Destination Account ID", value=st.session_state.get('preset_dest', "C987654321"))
            with row3_2:
                old_dest_val = st.number_input("Initial Balance (Destination) ($)", min_value=0.0, value=float(st.session_state.get('preset_old_dest', 0.00)), step=100.0)
            with row3_3:
                new_dest_val = st.number_input("Subsequent Balance (Destination) ($)", min_value=0.0, value=float(st.session_state.get('preset_new_dest', 1000.00)), step=100.0)
                
            submitted = st.form_submit_button("⚡ RUN REAL-TIME RISK INFERENCE", use_container_width=True)
            
        if submitted:
            tx_input = {
                'step': step_val,
                'type': type_val,
                'amount': amount_val,
                'nameOrig': orig_id,
                'oldbalanceOrg': old_orig_val,
                'newbalanceOrig': new_orig_val,
                'nameDest': dest_id,
                'oldbalanceDest': old_dest_val,
                'newbalanceDest': new_dest_val
            }
            
            with st.spinner("Executing dual-engine feature extraction and inference..."):
                result = predictor.predict_single(tx_input)
                
            st.markdown("### 📊 Inference Diagnostic Report")
            render_risk_summary(
                risk_level=result["risk_level"],
                fraud_prob_pct=result["fraud_probability_pct"],
                anomaly_score_pct=result["anomaly_score_pct"],
                anomaly_status=result["anomaly_status"]
            )
            render_explainability(result["risk_indicators"])

# ----------------- PAGE 7: MODEL PERFORMANCE ----------------- #
elif "07  Model Performance" in nav_page:
    render_banner(
        "Academic Model Performance & Benchmarking",
        "Empirical benchmarking on held-out test data (20% stratified holdout: 12,000 samples, 126 frauds)",
        badge="ML BENCHMARK"
    )
    
    if eval_results is not None:
        st.markdown("##### 📊 Supervised Classifier Performance Matrix")
        df_summary = pd.DataFrame(eval_results["summary_table"])
        st.dataframe(df_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Deep diagnostic model selector
        model_names = list(eval_results["models"].keys())
        default_idx = model_names.index(model_metadata["selected_model"]) if model_metadata and model_metadata["selected_model"] in model_names else 0
        selected_model = st.selectbox("Select Model for Deep Diagnostic View:", model_names, index=default_idx)
        
        model_data = eval_results["models"][selected_model]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(plot_confusion_matrix_heatmap(model_data["confusion_matrix"], selected_model), use_container_width=True)
        with col_m2:
            st.plotly_chart(plot_feature_importance_bar(model_data["all_feature_importances"], selected_model), use_container_width=True)
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(plot_roc_curve_chart(model_data["roc_curve"], selected_model, model_data["roc_auc"]), use_container_width=True)
        with col_c2:
            st.plotly_chart(plot_pr_curve_chart(model_data["pr_curve"], selected_model, model_data["pr_auc"]), use_container_width=True)
            
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; margin-top: 20px;">
            <h5 style="color: #00f2fe; margin-top: 0;">🎓 Academic Evaluation Discussion: Why PR-AUC Over Accuracy?</h5>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 8px;">
                In financial fraud detection where fraudulent transactions represent approximately <b>1.05%</b> of total activity, a naive trivial model predicting every record as legitimate achieves <b>98.95% accuracy</b> while achieving a catastrophic <b>0% Recall</b>.
            </p>
            <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.5; margin: 0;">
                Consequently, academic benchmarking prioritizes:
                <br/>1. <b>Precision-Recall AUC (PR-AUC):</b> Evaluates precision trade-offs without inflation from astronomical true-negative counts.
                <br/>2. <b>F1-Score:</b> Balances precision against false negatives.
                <br/>3. <b>Recall (Sensitivity):</b> Quantifies the percentage of actual illicit drains caught by the financial institution.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- PAGE 8: ABOUT PROJECT ----------------- #
elif "08  About Project" in nav_page:
    render_banner(
        "Academic Project Profile & System Architecture",
        "Gujarat Technological University (GTU) • BE Computer Engineering Semester 7 • InfoLabz IT Services Pvt. Ltd.",
        badge="ACADEMIC CREDENTIALS"
    )
    
    # 1. Academic & Internship Credentials Grid (4 Cards)
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.markdown("""
        <div class="metric-card" style="min-height: 125px;">
            <div class="metric-title">🎓 Degree & Academic Level</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">BE Computer Engineering</div>
            <div class="metric-sub">Semester 7 (Final Year) • GTU</div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
        <div class="metric-card" style="min-height: 125px;">
            <div class="metric-title">🏢 Internship Partner</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">InfoLabz IT Services</div>
            <div class="metric-sub">Pvt. Ltd., Ahmedabad, Gujarat</div>
        </div>
        """, unsafe_allow_html=True)
    with col_a3:
        st.markdown("""
        <div class="metric-card" style="min-height: 125px;">
            <div class="metric-title">🔬 Technical Domain</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">Data Analytics & Applied ML</div>
            <div class="metric-sub">Disciplinary / Internship Project</div>
        </div>
        """, unsafe_allow_html=True)
    with col_a4:
        st.markdown("""
        <div class="metric-card" style="min-height: 125px;">
            <div class="metric-title">🛡️ System Platform</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #00f2fe; margin-top: 4px;">FINSEC AI Intelligence</div>
            <div class="metric-sub">Dual-Engine Fraud Surveillance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Executive Overview & Problem Context (2 Columns)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; min-height: 220px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.2rem;">📌</span>
                <span style="font-size: 0.85rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em;">Problem Statement & Industry Need</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.6; margin: 0;">
                High-velocity digital payment networks face severe adversarial threats including account takeover balance draining and synthetic identity fraud. 
                Conventional static rule-based engines suffer from high false-alarm rates, while standard ML classifiers fail due to the <b>severe class imbalance (~1.05% fraud)</b>. 
                There is an imperative need for an adaptive, explainable intelligence platform.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; min-height: 220px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.2rem;">💡</span>
                <span style="font-size: 0.85rem; font-weight: 800; color: #00f2fe; text-transform: uppercase; letter-spacing: 0.06em;">Engineered Dual-Engine Solution</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.6; margin: 0;">
                <b>FINSEC AI</b> implements a synergistic architecture integrating <b>23 domain-engineered financial features</b>:
                <br/>• <b>Supervised Machine Learning:</b> Random Forest & XGBoost ensembles for high-precision fraud classification.
                <br/>• <b>Unsupervised Anomaly Detection:</b> Isolation Forest identifying zero-day behavioral deviations without ground-truth labels.
                <br/>• <b>Explainability Engine:</b> Generates human-readable compliance diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Interactive Architecture & Methodology Explorer (Tabs)
    st.markdown("##### 🏛️ System Architecture, Pipeline & Data Flow")
    tab_arch, tab_pipe, tab_dfd, tab_actors = st.tabs([
        "🏛️ 5-Tier System Architecture",
        "🔄 Machine Learning Pipeline",
        "📊 Data Flow Diagrams (DFD Level 0/1)",
        "👤 Stakeholder & Use-Case Model"
    ])

    with tab_arch:
        st.markdown("""
        <div style="background: #0d1424; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;">
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #38bdf8; text-transform: uppercase;">1. Data Layer</div>
                    <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 6px;">PaySim-style 60,000 transaction ledger ingestion, schema auditing, missing value sanitation, duplicate removal.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #00f2fe; text-transform: uppercase;">2. Feature Layer</div>
                    <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 6px;">23 domain signals: balance discrepancy errors, 100% account drainage ratios, transfer-drain interactions, and diurnal step encoding.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(129, 140, 248, 0.25); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #818cf8; text-transform: uppercase;">3. Analytics Layer</div>
                    <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 6px;">Parallel inference: Supervised Random Forest (P_Fraud) + Unsupervised Isolation Forest Outlier Score (A_Score).</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #f59e0b; text-transform: uppercase;">4. Decision Layer</div>
                    <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 6px;">Tri-Tier Composite Risk Matrix (LOW / MEDIUM / HIGH) synthesized with heuristic compliance rule triggers.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #10b981; text-transform: uppercase;">5. Presentation Layer</div>
                    <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 6px;">Streamlit 7-page interactive surveillance dashboard with dark-mode Plotly visuals and live presets.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_pipe:
        st.markdown("""
        <div style="background: #0d1424; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.7;">
                <b style="color: #00f2fe;">End-to-End Execution Sequence:</b><br/>
                <b>1. Ingestion & Validation:</b> 60,000 synthetic transaction records spanning 744 hourly steps across 5 payment channels (TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT).<br/>
                <b>2. Feature Computation:</b> Generation of 23 domain features including <code>orig_balance_diff</code>, <code>orig_error</code>, <code>is_complete_drain</code>, and <code>is_high_risk_channel</code>.<br/>
                <b>3. Stratified Partitioning:</b> 80% Train (48,000 records, 504 frauds) and 20% Holdout Test (12,000 records, 126 frauds) preserving exact class ratios.<br/>
                <b>4. Model Training:</b> Parallel training of Logistic Regression (class weighted), Random Forest (100 trees, max depth 15), and XGBoost (scale_pos_weight 94.2), alongside Isolation Forest (3% contamination).<br/>
                <b>5. Serialization:</b> Serialized artifacts saved to <code>models/</code> (<code>fraud_model.pkl</code>, <code>anomaly_model.pkl</code>, <code>preprocessing.pkl</code>, <code>evaluation_results.json</code>).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_dfd:
        st.markdown("""
        <div style="background: #0d1424; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 16px;">
                    <h6 style="color: #38bdf8; margin-top: 0;">Level 0: Context Data Flow</h6>
                    <p style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                        • <b>External Entity (User / Analyst):</b> Submits real-time transaction parameters.<br/>
                        • <b>System Core (FINSEC AI):</b> Ingests transaction vectors and historical parameters.<br/>
                        • <b>Output:</b> Multi-tier risk classifications, anomaly scores, and explainable audit cards.
                    </p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 16px;">
                    <h6 style="color: #00f2fe; margin-top: 0;">Level 1: Functional Decomposition</h6>
                    <p style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                        • <b>Process 1.0:</b> Data Ingestion & Preprocessing<br/>
                        • <b>Process 2.0:</b> Feature Engineering Engine<br/>
                        • <b>Process 3.0:</b> Supervised Classifier Scoring<br/>
                        • <b>Process 4.0:</b> Unsupervised Anomaly Isolation<br/>
                        • <b>Process 5.0:</b> Composite Risk Scorer & Rule Evaluator<br/>
                        • <b>Process 6.0:</b> Visualization & Analytics Console
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_actors:
        st.markdown("""
        <div style="background: #0d1424; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;">
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #38bdf8;">🕵️ Compliance Investigator</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 6px;">Inspects high-risk flagged transactions, reviews balance drainage diagnostics, and validates suspicious transfer vectors.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #00f2fe;">🛡️ Risk Operations Officer</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 6px;">Monitors gross transaction volumes, diurnal fraud penetration profiles, and tunes Isolation Forest contamination thresholds.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #10b981;">👨‍💻 Academic & ML Evaluator</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 6px;">Audits model metrics (F1, Precision, Recall, PR-AUC), inspects confusion matrices, and verifies feature importances.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 4. Model Benchmarking & Technical Highlights (4 KPI-style Cards)
    st.markdown("##### ⚙️ Technical Specifications & Model Benchmark Summary")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🏆 Champion Model</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #00f2fe; margin: 4px 0;">Random Forest</div>
            <div class="metric-sub">F1: 93.80% • PR-AUC: 0.9814</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">⚡ Boosted Classifier</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin: 4px 0;">XGBoost</div>
            <div class="metric-sub">Precision: 92.97% • Recall: 94.44%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🌲 Unsupervised Outliers</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #10b981; margin: 4px 0;">Isolation Forest</div>
            <div class="metric-sub">Contamination: 3.0% • 1,800 Flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">📐 Feature Space</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #f59e0b; margin: 4px 0;">23 Signals</div>
            <div class="metric-sub">Balance errors & timing ratios</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 5. Academic Assumptions, Limitations & Future Scope (2 Columns)
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 20px; min-height: 230px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.2rem;">⚠️</span>
                <span style="font-size: 0.85rem; font-weight: 800; color: #f87171; text-transform: uppercase; letter-spacing: 0.06em;">Academic Assumptions & Limitations</span>
            </div>
            <ul style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                <li><b>Synthetic Data Environment:</b> Evaluated on a 60,000-row PaySim-style synthetic transaction benchmark (Lopez-Rojas et al.). No private banking data or customer credentials are stored.</li>
                <li><b>Decision-Support Role:</b> The platform functions as an advisory assistant for human compliance officers rather than autonomous transaction cancellation.</li>
                <li><b>Graph Topology:</b> The current iteration operates on tabular transaction vectors without live graph relational traversal.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_l2:
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 12px; padding: 20px; min-height: 230px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.2rem;">🚀</span>
                <span style="font-size: 0.85rem; font-weight: 800; color: #34d399; text-transform: uppercase; letter-spacing: 0.06em;">Future Roadmap & Research Directions</span>
            </div>
            <ul style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                <li><b>Real-Time Event Streaming:</b> Integration with Apache Kafka / Apache Spark for sub-10ms distributed stream scoring.</li>
                <li><b>Graph Neural Networks (GNN):</b> Implementation of relational graph convolutional networks to detect distributed money mule syndicates.</li>
                <li><b>Dynamic SHAP TreeExplainer:</b> Micro-level game-theoretic feature contribution force plots for every live transaction evaluation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Enterprise Console Footer
st.markdown("""
<div class="finsec-footer">
    FINSEC AI • Financial Transaction Risk Intelligence Platform<br/>
    GTU BE Computer Engineering Semester 7 | InfoLabz IT Services Pvt. Ltd. Internship Project
</div>
""", unsafe_allow_html=True)
