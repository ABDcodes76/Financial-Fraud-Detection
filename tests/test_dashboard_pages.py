import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_dataset, clean_data, get_eda_summary
from src.prediction import TransactionPredictor
from src.utils import load_json
from dashboard.components import (
    plot_transaction_types, plot_amount_distribution, plot_fraud_by_type,
    plot_hourly_trend, plot_confusion_matrix_heatmap, plot_roc_curve_chart,
    plot_pr_curve_chart, plot_feature_importance_bar, plot_anomaly_scatter
)

class TestDashboardPagesAndFeatures(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_dataset()
        cls.df, cls.clean_report = clean_data(cls.raw_df)
        cls.eval_results = load_json("evaluation_results.json")
        cls.meta = load_json("model_metadata.json")
        cls.anomaly_meta = load_json("anomaly_metadata.json")
        cls.predictor = TransactionPredictor()

    def test_page1_overview_components(self):
        """Test Page 1: Overview KPIs and chart renderers."""
        eda = get_eda_summary(self.df)
        self.assertGreater(eda["total_transactions"], 0)
        self.assertGreater(eda["fraud_count"], 0)
        
        # Test figures
        fig_types = plot_transaction_types(self.df)
        self.assertIsNotNone(fig_types)
        fig_hourly = plot_hourly_trend(self.df)
        self.assertIsNotNone(fig_hourly)

    def test_page2_analytics_and_filtering(self):
        """Test Page 2: Analytics, log/raw amount distributions, and dynamic filtering."""
        fig_log = plot_amount_distribution(self.df, log_scale=True)
        self.assertIsNotNone(fig_log)
        fig_raw = plot_amount_distribution(self.df, log_scale=False)
        self.assertIsNotNone(fig_raw)
        
        # Filter test: simulate user filtering on PAYMENT and CASH_OUT
        filtered = self.df[self.df['type'].isin(['PAYMENT', 'CASH_OUT'])]
        self.assertGreater(len(filtered), 0)
        
        # Edge test: filter with restrictive amount range
        edge_filtered = self.df[(self.df['amount'] >= 1000) & (self.df['amount'] <= 5000)]
        self.assertGreater(len(edge_filtered), 0)

    def test_page3_fraud_analysis(self):
        """Test Page 3: Fraud channel concentration and balance draining metrics."""
        fig_fraud = plot_fraud_by_type(self.df)
        self.assertIsNotNone(fig_fraud)
        
        fraud_df = self.df[self.df['isFraud'] == 1]
        self.assertGreater(len(fraud_df), 0)
        drain_rate = (fraud_df['newbalanceOrig'] == 0).mean()
        self.assertGreaterEqual(drain_rate, 0.50, "Most fraud transactions should drain accounts")

    def test_page4_anomaly_detection(self):
        """Test Page 4: Isolation Forest scatter visualization and metadata."""
        sample_df = self.df.sample(100, random_state=42).copy()
        sample_df['orig_balance_diff'] = sample_df['oldbalanceOrg'] - sample_df['newbalanceOrig']
        sample_df['anomaly_status'] = 'Normal'
        fig_anom = plot_anomaly_scatter(sample_df)
        self.assertIsNotNone(fig_anom)
        self.assertIn("total_anomalies", self.anomaly_meta)

    def test_page5_prediction_presets(self):
        """Test Page 5: Transaction Prediction for all 3 academic presets."""
        
        # Preset 1: Legitimate Payment
        p1 = {
            'step': 14,
            'type': 'PAYMENT',
            'amount': 85.50,
            'oldbalanceOrg': 3500.00,
            'newbalanceOrig': 3414.50,
            'nameDest': 'M987654321',
            'oldbalanceDest': 0.00,
            'newbalanceDest': 0.00
        }
        res1 = self.predictor.predict_single(p1)
        self.assertEqual(res1["risk_level"], "LOW", "Preset 1 must be LOW risk")
        self.assertFalse(res1["is_fraud_predicted"])
        self.assertEqual(res1["anomaly_status"], "Normal")
        
        # Preset 2: Account Drain Fraud
        p2 = {
            'step': 3,
            'type': 'TRANSFER',
            'amount': 350000.00,
            'oldbalanceOrg': 350000.00,
            'newbalanceOrig': 0.00,
            'nameDest': 'C112233445',
            'oldbalanceDest': 0.00,
            'newbalanceDest': 350000.00
        }
        res2 = self.predictor.predict_single(p2)
        self.assertEqual(res2["risk_level"], "HIGH", "Preset 2 must be HIGH risk")
        self.assertTrue(res2["is_fraud_predicted"])
        self.assertGreaterEqual(res2["fraud_probability"], 0.85)
        
        # Preset 3: Large Discrepancy Anomaly
        p3 = {
            'step': 42,
            'type': 'CASH_OUT',
            'amount': 180000.00,
            'oldbalanceOrg': 250000.00,
            'newbalanceOrig': 200000.00,
            'nameDest': 'C556677889',
            'oldbalanceDest': 10000.00,
            'newbalanceDest': 190000.00
        }
        res3 = self.predictor.predict_single(p3)
        self.assertIn(res3["risk_level"], ["MEDIUM", "HIGH"], "Preset 3 must be elevated risk")
        categories = [x["category"] for x in res3["risk_indicators"]]
        self.assertIn("Balance Discrepancy", categories)

    def test_page6_model_performance(self):
        """Test Page 6: Model performance benchmarking charts and confusion matrices."""
        models = self.eval_results["models"]
        for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            self.assertIn(m_name, models)
            m_data = models[m_name]
            
            # Test Confusion Matrix heatmap
            fig_cm = plot_confusion_matrix_heatmap(m_data["confusion_matrix"], m_name)
            self.assertIsNotNone(fig_cm)
            
            # Test ROC curve
            fig_roc = plot_roc_curve_chart(m_data["roc_curve"], m_name, m_data["roc_auc"])
            self.assertIsNotNone(fig_roc)
            
            # Test PR curve
            fig_pr = plot_pr_curve_chart(m_data["pr_curve"], m_name, m_data["pr_auc"])
            self.assertIsNotNone(fig_pr)
            
            # Test Feature Importance
            fig_feat = plot_feature_importance_bar(m_data["all_feature_importances"], m_name)
            self.assertIsNotNone(fig_feat)

if __name__ == "__main__":
    unittest.main()
