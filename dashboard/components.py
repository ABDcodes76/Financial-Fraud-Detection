import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

DARK_THEME_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.5)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
    margin=dict(t=45, b=25, l=25, r=25)
)

def render_banner(title: str, subtitle: str, badge: str = "FINSEC AI CORE"):
    """Renders a top enterprise FinTech header banner."""
    st.markdown(f"""
    <div class="academic-banner">
        <div>
            <div class="academic-title">{title}</div>
            <div class="academic-subtitle">{subtitle}</div>
        </div>
        <div class="academic-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(title: str, value: str, subtitle: str = "", icon: str = "📈"):
    """Renders an enterprise KPI metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_risk_summary(risk_level: str, fraud_prob_pct: float, anomaly_score_pct: float, anomaly_status: str):
    """Renders an enterprise risk decision-support result panel."""
    badge_class = f"badge-{risk_level.lower()}"
    status_color = "#f87171" if anomaly_status == "Anomalous" else "#34d399"
    
    st.markdown(f"""
    <div class="risk-assessment-container">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 16px; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Assessed Risk Classification</span><br/>
                <span class="{badge_class}" style="margin-top: 6px;">{risk_level} RISK LEVEL</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Unsupervised Anomaly Flag</span><br/>
                <span style="font-size: 1.15rem; font-weight: 800; color: {status_color}; font-family: 'JetBrains Mono', monospace;">
                    {'⚠️ ' if anomaly_status == 'Anomalous' else '✓ '}{anomaly_status.upper()}
                </span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background: rgba(13, 20, 36, 0.7); padding: 16px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">Supervised Fraud Probability:</span>
                    <span style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{fraud_prob_pct:.1f}%</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.08); border-radius: 6px; height: 10px; width: 100%; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #38bdf8 0%, #ef4444 100%); height: 10px; border-radius: 6px; width: {min(100.0, max(0.0, fraud_prob_pct))}%;"></div>
                </div>
                <div style="font-size: 0.68rem; color: #64748b; margin-top: 6px;">Evaluated via ensemble decision boundary</div>
            </div>
            <div style="background: rgba(13, 20, 36, 0.7); padding: 16px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">Isolation Forest Outlier Score:</span>
                    <span style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{anomaly_score_pct:.1f}%</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.08); border-radius: 6px; height: 10px; width: 100%; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #34d399 0%, #f59e0b 100%); height: 10px; border-radius: 6px; width: {min(100.0, max(0.0, anomaly_score_pct))}%;"></div>
                </div>
                <div style="font-size: 0.68rem; color: #64748b; margin-top: 6px;">Geometric distance from routine transaction clusters</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_explainability(indicators: List[Dict[str, str]]):
    """Renders explainable fraud risk indicators."""
    st.markdown("##### 🛡️ Explainable Fraud Diagnostics & Decision Rationale")
    for ind in indicators:
        sev = ind.get("severity", "INFO").lower()
        box_class = f"indicator-{sev}"
        category = ind.get("category", "Indicator")
        msg = ind.get("message", "")
        badge_color = {
            "critical": "#ef4444",
            "high": "#f97316",
            "medium": "#f59e0b",
            "low": "#10b981",
            "info": "#38bdf8"
        }.get(sev, "#94a3b8")
        
        st.markdown(f"""
        <div class="indicator-box {box_class}">
            <span style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: {badge_color}; letter-spacing: 0.06em;">
                [{ind.get('severity', 'INFO')}] {category}
            </span>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-top: 4px; line-height: 1.4;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- PLOTLY CHART FUNCTIONS (DARK THEMED) ----------------- #

def plot_transaction_types(df: pd.DataFrame) -> go.Figure:
    counts = df['type'].value_counts().reset_index()
    counts.columns = ['type', 'count']
    
    colors = ['#00f2fe', '#38bdf8', '#818cf8', '#a855f7', '#ec4899']
    fig = px.pie(
        counts, values='count', names='type',
        title="Transaction Volume Breakdown by Channel",
        hole=0.55,
        color_discrete_sequence=colors
    )
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=360,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#0d121f', width=2))
    )
    return fig

def plot_amount_distribution(df: pd.DataFrame, log_scale: bool = True) -> go.Figure:
    amounts = df['amount'].clip(upper=df['amount'].quantile(0.99)) if not log_scale else np.log1p(df['amount'])
    x_label = "Log(1 + Amount)" if log_scale else "Transaction Amount ($)"
    
    fig = px.histogram(
        x=amounts,
        nbins=45,
        title=f"Transaction Distribution ({x_label})",
        labels={'x': x_label, 'y': 'Transaction Frequency'},
        color_discrete_sequence=['#00f2fe']
    )
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=360,
        bargap=0.08
    )
    fig.update_traces(marker=dict(line=dict(color='#0d1424', width=1)))
    return fig

def plot_fraud_by_type(df: pd.DataFrame) -> go.Figure:
    fraud_group = df.groupby('type')['isFraud'].agg(['count', 'sum']).reset_index()
    fraud_group['fraud_pct'] = (fraud_group['sum'] / fraud_group['count']) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fraud_group['type'],
        y=fraud_group['sum'],
        name="Fraud Incidents",
        marker=dict(
            color='#ef4444',
            line=dict(color='rgba(239, 68, 68, 0.4)', width=1)
        ),
        text=fraud_group['sum'],
        textposition='auto',
        textfont=dict(color='#ffffff', family='JetBrains Mono')
    ))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="Fraud Distribution Across Channels (Strictly TRANSFER & CASH_OUT)",
        xaxis_title="Transaction Channel Type",
        yaxis_title="Confirmed Fraud Cases",
        height=360
    )
    return fig

def plot_hourly_trend(df: pd.DataFrame) -> go.Figure:
    df_temp = df.copy()
    df_temp['hour'] = df_temp['step'] % 24
    hourly = df_temp.groupby('hour').agg(
        total_tx=('amount', 'count'),
        fraud_tx=('isFraud', 'sum')
    ).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly['hour'],
        y=hourly['total_tx'],
        name="Total Activity",
        marker=dict(color='rgba(56, 189, 248, 0.25)', line=dict(color='#38bdf8', width=1)),
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['fraud_tx'],
        name="Fraud Incidents",
        mode='lines+markers',
        line=dict(color='#ef4444', width=3),
        marker=dict(size=7, color='#ef4444', symbol='circle'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="24-Hour Diurnal Activity & Fraud Penetration Profile",
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode='linear', dtick=2, gridcolor="rgba(255, 255, 255, 0.06)"),
        yaxis=dict(title="Total Transactions", side='left', gridcolor="rgba(255, 255, 255, 0.06)"),
        yaxis2=dict(title="Fraud Incursions", overlaying='y', side='right', color='#ef4444', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    return fig

def plot_confusion_matrix_heatmap(cm_dict: Dict[str, Any], model_name: str) -> go.Figure:
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]
    
    matrix = [[tn, fp], [fn, tp]]
    labels = [["True Neg (Normal)", "False Pos (False Alarm)"],
              ["False Neg (Missed Fraud)", "True Pos (Caught Fraud)"]]
    
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{matrix[i][j]:,}</b><br><span style='font-size:10px;'>{labels[i][j]}</span>",
                font=dict(color='white' if matrix[i][j] > (tn / 2) else '#94a3b8', size=13),
                showarrow=False
            ))
            
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Predicted Normal", "Predicted Fraud"],
        y=["Actual Normal", "Actual Fraud"],
        colorscale=[[0, '#0a1120'], [0.2, '#0369a1'], [1, '#00f2fe']],
        showscale=False
    ))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=f"Confusion Matrix: {model_name}",
        annotations=annotations,
        height=350
    )
    return fig

def plot_roc_curve_chart(roc_dict: Dict[str, Any], model_name: str, roc_auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roc_dict["fpr"],
        y=roc_dict["tpr"],
        mode='lines',
        name=f"{model_name} (AUC = {roc_auc:.4f})",
        line=dict(color='#00f2fe', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name="Random Chance Baseline",
        line=dict(color='#64748b', dash='dash')
    ))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=f"Receiver Operating Characteristic (ROC): {model_name}",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.5, y=0.1, bgcolor='rgba(15, 23, 42, 0.8)'),
        height=350
    )
    return fig

def plot_pr_curve_chart(pr_dict: Dict[str, Any], model_name: str, pr_auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pr_dict["recall"],
        y=pr_dict["precision"],
        mode='lines',
        name=f"{model_name} (PR-AUC = {pr_auc:.4f})",
        line=dict(color='#10b981', width=3)
    ))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=f"Precision-Recall Curve: {model_name}",
        xaxis_title="Recall (Fraud Coverage)",
        yaxis_title="Precision (Flag Reliability)",
        legend=dict(x=0.05, y=0.1, bgcolor='rgba(15, 23, 42, 0.8)'),
        height=350
    )
    return fig

def plot_feature_importance_bar(feat_dict: Dict[str, float], model_name: str) -> go.Figure:
    items = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    df_feat = pd.DataFrame(items, columns=['Feature', 'Importance']).iloc[::-1]
    
    fig = px.bar(
        df_feat, x='Importance', y='Feature', orientation='h',
        title=f"Top 10 Feature Importances ({model_name})",
        color='Importance',
        color_continuous_scale=[[0, '#0284c7'], [1, '#00f2fe']]
    )
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=350,
        coloraxis_showscale=False
    )
    return fig

def plot_anomaly_scatter(df_sample: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df_sample,
        x='amount',
        y='orig_balance_diff',
        color='anomaly_status',
        color_discrete_map={'Normal': '#38bdf8', 'Anomalous': '#ef4444'},
        hover_data=['type', 'step', 'isFraud'],
        title="Isolation Forest Multivariate Scatter: Transaction Amount vs Origin Balance Change",
        labels={'amount': 'Transaction Amount ($)', 'orig_balance_diff': 'Origin Balance Depletion ($)'}
    )
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=390,
        legend=dict(bgcolor='rgba(15, 23, 42, 0.8)')
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    return fig
