import io
import datetime
import uuid
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from src.dataset_validator import ValidationResult

class FinsecReportPDF(FPDF):
    """Custom stylized PDF report generator for FINSEC AI audit reports."""
    
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        self.report_id = f"FINSEC-{uuid.uuid4().hex[:8].upper()}"
        self.generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    def header(self):
        # Top Header Bar
        self.set_fill_color(15, 23, 42)  # Dark slate
        self.rect(0, 0, 210, 18, 'F')
        
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(56, 189, 248)  # Cyan
        self.set_xy(12, 5)
        self.cell(100, 8, 'FINSEC AI | ENTERPRISE FRAUD AUDIT REPORT', new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        self.set_font('Helvetica', '', 8)
        self.set_text_color(148, 163, 184)  # Light slate
        self.set_xy(110, 5)
        self.cell(88, 8, f"DOC ID: {self.report_id}  |  {self.generated_at}", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(16)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(100, 8, 'CONFIDENTIAL -- FOR FINANCIAL AUDIT & SURVEILLANCE PURPOSES ONLY', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(86, 8, f'Page {self.page_no()}', new_x=XPos.RIGHT, new_y=YPos.TOP)

    def chapter_title(self, num: int, title: str):
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(241, 245, 249)
        self.set_text_color(15, 23, 42)
        self.cell(186, 6.5, f" {num}. {title.upper()}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def chapter_subtitle(self, text: str):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(186, 4, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)


def generate_pdf_report(
    analyzed_df: pd.DataFrame,
    validation_result: ValidationResult,
    metrics: Dict[str, Any]
) -> bytes:
    """
    Generates a production-grade forensic audit PDF report in memory.
    """
    pdf = FinsecReportPDF()
    pdf.add_page()
    
    # ----------------- SECTION 1: REPORT TITLE BANNER ----------------- #
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(186, 7, "Financial Transaction Forensic Audit & Risk Intelligence Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(186, 4.5, "Automated Dual-Engine ML Surveillance Audit: Random Forest Supervised + Isolation Forest Anomaly Detection", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ----------------- SECTION 2: DATASET INGESTION & QUALITY ----------------- #
    pdf.chapter_title(1, "Dataset Ingestion & Quality Audit")
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(30, 41, 59)
    
    dup_str = f"{validation_result.duplicate_count:,} duplicate(s) sanitized" if validation_result.duplicate_count > 0 else "0 duplicates (clean)"
    null_str = f"{sum(validation_result.null_counts.values()):,} missing values imputed" if validation_result.null_counts else "0 missing values"
    gt_str = "Present (Ground Truth Benchmarking Enabled)" if validation_result.has_ground_truth else "Not Provided (Inference Surveillance Mode)"

    pdf.cell(93, 4.5, f"- Ingested Records: {metrics.get('total_records', len(analyzed_df)):,}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(93, 4.5, f"- Schema Validation: Passed (7 required columns verified)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(93, 4.5, f"- Deduplication: {dup_str}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(93, 4.5, f"- Data Completeness: {null_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(186, 4.5, f"- Target Labels (isFraud): {gt_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ----------------- SECTION 3: EXECUTIVE SUMMARY KPIS ----------------- #
    pdf.chapter_title(2, "Executive Surveillance Metrics & Exposure Summary")
    
    # Draw KPI summary grid
    tot_vol = metrics.get('total_volume', 0.0)
    pred_fraud_cnt = metrics.get('predicted_fraud_count', 0)
    pred_fraud_pct = metrics.get('predicted_fraud_pct', 0.0)
    pred_fraud_vol = metrics.get('predicted_fraud_volume', 0.0)
    anom_cnt = metrics.get('anomalies_count', 0)
    anom_pct = metrics.get('anomalies_pct', 0.0)
    high_cnt = metrics.get('high_risk_count', 0)
    high_pct = metrics.get('high_risk_pct', 0.0)
    high_vol = metrics.get('high_risk_volume', 0.0)

    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    
    # Row 1 of KPI boxes
    y_pos = pdf.get_y()
    pdf.rect(12, y_pos, 58, 13, 'DF')
    pdf.rect(76, y_pos, 58, 13, 'DF')
    pdf.rect(140, y_pos, 58, 13, 'DF')
    
    # Box 1
    pdf.set_xy(14, y_pos + 1.5)
    pdf.set_font('Helvetica', 'B', 6.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(54, 3, "TOTAL AUDITED VOLUME", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(14, y_pos + 5.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(54, 5.5, f"${tot_vol:,.2f}", new_x=XPos.RIGHT, new_y=YPos.TOP)

    # Box 2
    pdf.set_xy(78, y_pos + 1.5)
    pdf.set_font('Helvetica', 'B', 6.5)
    pdf.set_text_color(220, 38, 38)  # Red
    pdf.cell(54, 3, "PREDICTED FRAUD DETECTIONS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(78, y_pos + 5.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(54, 5.5, f"{pred_fraud_cnt:,} ({pred_fraud_pct:.2f}%)", new_x=XPos.RIGHT, new_y=YPos.TOP)

    # Box 3
    pdf.set_xy(142, y_pos + 1.5)
    pdf.set_font('Helvetica', 'B', 6.5)
    pdf.set_text_color(2, 132, 199)  # Blue
    pdf.cell(54, 3, "BEHAVIORAL ANOMALIES", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(142, y_pos + 5.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(54, 5.5, f"{anom_cnt:,} ({anom_pct:.2f}%)", new_x=XPos.RIGHT, new_y=YPos.TOP)

    # Row 2 of KPI boxes
    y_pos2 = y_pos + 15
    pdf.rect(12, y_pos2, 90, 13, 'DF')
    pdf.rect(108, y_pos2, 90, 13, 'DF')

    # Box 4
    pdf.set_xy(14, y_pos2 + 1.5)
    pdf.set_font('Helvetica', 'B', 6.5)
    pdf.set_text_color(220, 38, 38)
    pdf.cell(86, 3, "HIGH-RISK EXPOSURE TIER", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(14, y_pos2 + 5.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(86, 5.5, f"{high_cnt:,} Tx ({high_pct:.1f}%) | ${high_vol:,.2f}", new_x=XPos.RIGHT, new_y=YPos.TOP)

    # Box 5
    pdf.set_xy(110, y_pos2 + 1.5)
    pdf.set_font('Helvetica', 'B', 6.5)
    pdf.set_text_color(217, 119, 6)  # Amber
    pdf.cell(86, 3, "FRAUD CURRENCY AT RISK", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(110, y_pos2 + 5.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(86, 5.5, f"${pred_fraud_vol:,.2f} of total volume", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_xy(12, y_pos2 + 16)
    pdf.ln(1)

    # ----------------- SECTION 4: RISK STRATIFICATION ----------------- #
    pdf.chapter_title(3, "Multi-Tier Risk Stratification Breakdown")
    
    # Table Header
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.cell(38, 5.5, "Risk Tier", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(32, 5.5, "Transaction Count", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(28, 5.5, "% of Ledger", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(42, 5.5, "Financial Volume", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(46, 5.5, "Recommended Action", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Rows
    med_cnt = metrics.get('medium_risk_count', 0)
    med_pct = metrics.get('medium_risk_pct', 0.0)
    med_vol = float(analyzed_df[analyzed_df['risk_level'] == 'MEDIUM']['amount'].sum()) if len(analyzed_df) > 0 else 0.0

    low_cnt = metrics.get('low_risk_count', 0)
    low_pct = metrics.get('low_risk_pct', 0.0)
    low_vol = float(analyzed_df[analyzed_df['risk_level'] == 'LOW']['amount'].sum()) if len(analyzed_df) > 0 else 0.0

    tiers = [
        ("HIGH RISK", high_cnt, high_pct, high_vol, "Immediate Freeze / Step-Up 2FA", (254, 242, 242), (185, 28, 28)),
        ("MEDIUM RISK", med_cnt, med_pct, med_vol, "Secondary AML Verification", (255, 251, 235), (180, 83, 9)),
        ("LOW RISK", low_cnt, low_pct, low_vol, "Standard Automated Settlement", (240, 253, 244), (21, 128, 61))
    ]

    for name, cnt, pct, vol, action, fill_rgb, text_rgb in tiers:
        pdf.set_fill_color(*fill_rgb)
        pdf.set_text_color(*text_rgb)
        pdf.set_font('Helvetica', 'B', 7.5)
        pdf.cell(38, 5, f"  {name}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.cell(32, 5, f"{cnt:,}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(28, 5, f"{pct:.2f}%", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(42, 5, f"${vol:,.2f}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(46, 5, action, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)

    # ----------------- SECTION 5: CHANNEL VECTOR ANALYSIS ----------------- #
    pdf.chapter_title(4, "Transaction Channel Vector Analysis")
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.cell(36, 5.5, "Payment Channel", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(35, 5.5, "Volume ($)", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(35, 5.5, "Total Count", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(40, 5.5, "Predicted Fraud Tx", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(40, 5.5, "Channel Fraud Rate", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    channel_sum = metrics.get('channel_summary', {})
    pdf.set_text_color(30, 41, 59)
    pdf.set_font('Helvetica', '', 7.5)
    
    for t_type, ch in channel_sum.items():
        is_highlight = ch['fraud_count'] > 0
        if is_highlight:
            pdf.set_fill_color(254, 242, 242)
            pdf.set_text_color(185, 28, 28)
            pdf.set_font('Helvetica', 'B', 7.5)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(51, 65, 85)
            pdf.set_font('Helvetica', '', 7.5)

        pdf.cell(36, 5, f"  {t_type}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(35, 5, f"${ch['volume']:,.2f}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(35, 5, f"{ch['count']:,}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(40, 5, f"{ch['fraud_count']:,}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(40, 5, f"{ch['fraud_rate']:.2f}%", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)

    # ----------------- SECTION 6: GROUND TRUTH BENCHMARK (IF AVAILABLE) ----------------- #
    if validation_result.has_ground_truth and "evaluation" in metrics:
        eval_data = metrics["evaluation"]
        pdf.chapter_title(5, "Supervised Ground Truth Verification & Evaluation Benchmark")
        
        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(30, 41, 59)
        cm = eval_data["confusion_matrix"]
        
        pdf.cell(93, 4.5, f"- Accuracy: {eval_data['accuracy']*100:.2f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(93, 4.5, f"- True Positives (TP): {cm['tp']:,}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(93, 4.5, f"- Precision: {eval_data['precision']*100:.2f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(93, 4.5, f"- False Positives (FP): {cm['fp']:,}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(93, 4.5, f"- Recall / Sensitivity: {eval_data['recall']*100:.2f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(93, 4.5, f"- True Negatives (TN): {cm['tn']:,}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(93, 4.5, f"- F1-Score: {eval_data['f1_score']:.4f}", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(93, 4.5, f"- False Negatives (FN): {cm['fn']:,}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(186, 4.5, f"- ROC-AUC Score: {eval_data['roc_auc']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # ----------------- SECTION 7: HIGH-RISK FLAGGED FORENSIC SAMPLE ----------------- #
    sample_title_num = 6 if (validation_result.has_ground_truth and "evaluation" in metrics) else 5
    pdf.chapter_title(sample_title_num, "Forensic Audit Sample of High-Risk Transactions")
    
    flagged_tx = analyzed_df.sort_values(by=['fraud_probability', 'risk_score'], ascending=False).head(5)
    
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 7)
    pdf.cell(16, 5, "Step / ID", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(20, 5, "Channel", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(28, 5, "Amount ($)", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(20, 5, "Fraud Prob", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(18, 5, "Risk Tier", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(84, 5, "Forensic Indicator Flag", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('Helvetica', '', 7)
    for idx, row in flagged_tx.iterrows():
        is_hr = row['risk_level'] == 'HIGH'
        pdf.set_fill_color(254, 242, 242) if is_hr else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(185, 28, 28) if is_hr else pdf.set_text_color(51, 65, 85)
        
        tx_label = f"#{idx+1}"
        pdf.cell(16, 4.5, tx_label, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(20, 4.5, str(row['type']), fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(28, 4.5, f"${row['amount']:,.2f}", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(20, 4.5, f"{row['fraud_probability']*100:.1f}%", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(18, 4.5, str(row['risk_level']), fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        indicator_text = str(row['primary_indicator'])[:50]
        pdf.cell(84, 4.5, f" {indicator_text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)

    # ----------------- SECTION 8: LEGAL & COMPLIANCE DISCLAIMER ----------------- #
    disc_num = sample_title_num + 1
    pdf.chapter_title(disc_num, "Compliance Certification & Legal Disclaimer")
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(100, 116, 139)
    disclaimer_text = (
        "This forensic risk audit was automatically produced by FINSEC AI utilizing dual-engine Machine Learning "
        "(Supervised Random Forest Classifier + Unsupervised Isolation Forest Anomaly Detection). "
        "Outputs are provided to assist certified AML/Fraud Compliance Officers and do not constitute an autonomous "
        "legal conviction. All flagged transactions above $200,000 or marked HIGH risk must undergo mandatory human verification."
    )
    pdf.multi_cell(186, 3.5, disclaimer_text)

    # Return PDF byte stream
    return bytes(pdf.output())
