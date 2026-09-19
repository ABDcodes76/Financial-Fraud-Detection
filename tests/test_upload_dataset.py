import io
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_validator import (
    validate_dataset, generate_sample_template, REQUIRED_COLUMNS, ValidationResult
)
from src.bulk_prediction import BulkPredictionEngine
from src.report_generator import generate_pdf_report

class TestUploadAndAnalyzeDatasetFeature(unittest.TestCase):
    """
    Unit and integration test suite for the Upload & Analyze Dataset feature.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = BulkPredictionEngine()
        cls.sample_df = generate_sample_template()

    def test_01_sample_template_generation(self):
        """Verify sample template contains required schema and realistic rows."""
        self.assertIsInstance(self.sample_df, pd.DataFrame)
        self.assertEqual(len(self.sample_df), 10)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.sample_df.columns)
        self.assertIn('isFraud', self.sample_df.columns)

    def test_02_validation_success_valid_df(self):
        """Verify validation passes for canonical schema DataFrame."""
        result = validate_dataset(self.sample_df)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.row_count, 10)
        self.assertEqual(len(result.error_messages), 0)
        self.assertTrue(result.has_ground_truth)
        self.assertIsNotNone(result.cleaned_df)

    def test_03_validation_missing_columns(self):
        """Verify validation fails when mandatory columns are omitted."""
        invalid_df = self.sample_df.drop(columns=['amount', 'oldbalanceOrg'])
        result = validate_dataset(invalid_df)
        self.assertFalse(result.is_valid)
        self.assertIn('amount', result.missing_required)
        self.assertIn('oldbalanceOrg', result.missing_required)
        self.assertGreater(len(result.error_messages), 0)

    def test_04_validation_invalid_transaction_types(self):
        """Verify validation catches unauthorized payment channel types."""
        invalid_df = self.sample_df.copy()
        invalid_df.loc[0, 'type'] = 'BITCOIN_TRANSFER'
        result = validate_dataset(invalid_df)
        self.assertFalse(result.is_valid)
        self.assertIn('BITCOIN_TRANSFER', result.invalid_types)

    def test_05_validation_csv_bytes_and_string_io(self):
        """Verify validator handles file byte streams and CSV text buffers."""
        csv_text = self.sample_df.to_csv(index=False)
        csv_bytes = csv_text.encode('utf-8')
        
        # Test io.BytesIO
        res_bytes = validate_dataset(io.BytesIO(csv_bytes))
        self.assertTrue(res_bytes.is_valid)
        
        # Test io.StringIO
        res_str = validate_dataset(io.StringIO(csv_text))
        self.assertTrue(res_str.is_valid)

    def test_06_validation_empty_input(self):
        """Verify validator gracefully handles empty files."""
        res_empty = validate_dataset(pd.DataFrame())
        self.assertFalse(res_empty.is_valid)
        self.assertIn("empty", res_empty.error_messages[0].lower())

    def test_07_bulk_prediction_output_schema(self):
        """Verify batch prediction produces all expected columns and valid values."""
        val = validate_dataset(self.sample_df)
        analyzed = self.engine.analyze_dataset(val.cleaned_df)
        
        expected_cols = [
            'fraud_probability', 'fraud_probability_pct', 'fraud_prediction',
            'is_fraud_predicted', 'anomaly_score', 'anomaly_score_pct',
            'anomaly_status', 'is_anomaly', 'risk_level', 'risk_score',
            'primary_indicator'
        ]
        for col in expected_cols:
            self.assertIn(col, analyzed.columns, f"Missing output column {col}")

        # Assert probability values are bounded [0, 1]
        self.assertTrue((analyzed['fraud_probability'] >= 0.0).all())
        self.assertTrue((analyzed['fraud_probability'] <= 1.0).all())

        # Assert risk levels are in canonical set
        valid_tiers = {'HIGH', 'MEDIUM', 'LOW'}
        self.assertTrue(set(analyzed['risk_level'].unique()).issubset(valid_tiers))

    def test_08_bulk_prediction_without_labels(self):
        """Verify inference operates smoothly when ground-truth labels are absent."""
        unlabeled_df = self.sample_df.drop(columns=['isFraud', 'isFlaggedFraud'])
        val = validate_dataset(unlabeled_df)
        self.assertFalse(val.has_ground_truth)
        
        analyzed = self.engine.analyze_dataset(val.cleaned_df)
        kpis = self.engine.compute_summary_kpis(analyzed)
        
        self.assertFalse(kpis['has_ground_truth'])
        self.assertNotIn('evaluation', kpis)
        self.assertGreater(kpis['total_records'], 0)

    def test_09_ground_truth_evaluation_metrics(self):
        """Verify evaluation metrics are accurately calculated when isFraud is present."""
        val = validate_dataset(self.sample_df)
        analyzed = self.engine.analyze_dataset(val.cleaned_df)
        kpis = self.engine.compute_summary_kpis(analyzed)
        
        self.assertTrue(kpis['has_ground_truth'])
        self.assertIn('evaluation', kpis)
        ev = kpis['evaluation']
        self.assertIn('accuracy', ev)
        self.assertIn('precision', ev)
        self.assertIn('recall', ev)
        self.assertIn('f1_score', ev)
        self.assertIn('roc_auc', ev)
        self.assertIn('confusion_matrix', ev)
        self.assertGreaterEqual(ev['accuracy'], 0.0)
        self.assertLessEqual(ev['accuracy'], 1.0)

    def test_10_pdf_report_generation(self):
        """Verify PDF report generation produces valid, non-empty byte buffer."""
        val = validate_dataset(self.sample_df)
        analyzed = self.engine.analyze_dataset(val.cleaned_df)
        kpis = self.engine.compute_summary_kpis(analyzed)
        
        pdf_bytes = generate_pdf_report(analyzed, val, kpis)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_11_pdf_report_generation_unlabeled(self):
        """Verify PDF report generation succeeds for unlabeled datasets."""
        unlabeled_df = self.sample_df.drop(columns=['isFraud', 'isFlaggedFraud'])
        val = validate_dataset(unlabeled_df)
        analyzed = self.engine.analyze_dataset(val.cleaned_df)
        kpis = self.engine.compute_summary_kpis(analyzed)
        
        pdf_bytes = generate_pdf_report(analyzed, val, kpis)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

if __name__ == "__main__":
    unittest.main()
