import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_dataset, clean_data, REQUIRED_COLUMNS
from src.feature_engineering import engineer_features, prepare_single_transaction
from src.prediction import TransactionPredictor
from src.utils import get_data_path, load_json

class TestFinancialFraudPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up resources once before all tests."""
        cls.data_path = get_data_path("transactions.csv")
        cls.assertTrue(cls.data_path.exists(), "transactions.csv must exist before running tests")
        cls.df_raw = pd.read_csv(cls.data_path, nrows=500)
        cls.predictor = TransactionPredictor()

    def test_01_dataset_schema(self):
        """Verify that all required financial transaction columns exist."""
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.df_raw.columns, f"Required column '{col}' missing from dataset")

    def test_02_data_cleaning(self):
        """Verify data cleaning handles negatives, missing values, and duplicates."""
        df_clean, audit = clean_data(self.df_raw.copy())
        self.assertGreater(len(df_clean), 0)
        self.assertEqual(df_clean.isnull().sum().sum(), 0, "Cleaned dataset should have zero nulls")
        self.assertTrue((df_clean['amount'] >= 0).all(), "Amounts should all be non-negative")
        self.assertTrue((df_clean['oldbalanceOrg'] >= 0).all(), "Balances should all be non-negative")

    def test_03_feature_engineering_dimensions(self):
        """Verify engineered features matrix contains all required model inputs."""
        df_clean, _ = clean_data(self.df_raw.copy())
        X, y, features, scaler = engineer_features(df_clean, is_training=False)
        self.assertEqual(X.shape[0], len(df_clean))
        self.assertEqual(len(features), 23)
        self.assertIn('error_balance_orig', X.columns)
        self.assertIn('error_balance_dest', X.columns)
        self.assertIn('is_drained_orig', X.columns)
        self.assertIn('log_amount', X.columns)

    def test_04_prediction_normal_transaction(self):
        """Test inference on a typical legitimate merchant payment."""
        normal_tx = {
            'step': 14,
            'type': 'PAYMENT',
            'amount': 45.0,
            'nameOrig': 'C123456789',
            'oldbalanceOrg': 1500.0,
            'newbalanceOrig': 1455.0,
            'nameDest': 'M987654321',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        res = self.predictor.predict_single(normal_tx)
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM"])
        self.assertFalse(res["is_fraud_predicted"])
        self.assertGreaterEqual(res["fraud_probability"], 0.0)
        self.assertLessEqual(res["fraud_probability"], 1.0)
        self.assertGreaterEqual(len(res["risk_indicators"]), 1)

    def test_05_prediction_fraud_drain_transaction(self):
        """Test inference on a classic account drain takeover."""
        fraud_tx = {
            'step': 5,
            'type': 'TRANSFER',
            'amount': 400000.0,
            'nameOrig': 'C888888888',
            'oldbalanceOrg': 400000.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'C999999999',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 400000.0
        }
        res = self.predictor.predict_single(fraud_tx)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertTrue(res["is_fraud_predicted"])
        self.assertGreaterEqual(res["fraud_probability"], 0.70)
        
        # Check explainability indicators
        categories = [ind["category"] for ind in res["risk_indicators"]]
        self.assertIn("Balance Draining", categories)

    def test_06_model_evaluation_metrics_integrity(self):
        """Verify that saved metrics are real and within mathematical bounds."""
        eval_data = load_json("evaluation_results.json")
        self.assertIn("models", eval_data)
        self.assertIn("Random Forest", eval_data["models"])
        
        rf_metrics = eval_data["models"]["Random Forest"]
        self.assertGreaterEqual(rf_metrics["accuracy"], 0.90)
        self.assertGreaterEqual(rf_metrics["f1_score"], 0.70)
        self.assertGreaterEqual(rf_metrics["roc_auc"], 0.90)
        self.assertGreaterEqual(rf_metrics["pr_auc"], 0.80)

if __name__ == "__main__":
    unittest.main()
