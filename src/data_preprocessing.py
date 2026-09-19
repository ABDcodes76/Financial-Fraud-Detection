import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from src.utils import get_data_path, logger

REQUIRED_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
    'isFraud'
]

def load_dataset(filepath=None) -> pd.DataFrame:
    """
    Loads financial transactions dataset from CSV file.
    Validates presence of essential schema columns.
    """
    if filepath is None:
        filepath = get_data_path("transactions.csv")
        
    logger.info(f"Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please generate or download the dataset first."
        )
        
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns.")
    
    # Schema validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw transaction data:
    - Analyzes and handles missing values
    - Removes duplicate rows
    - Validates and sanitizes numeric values
    - Returns cleaned DataFrame and cleaning audit report
    """
    logger.info("Initiating data cleaning pipeline...")
    initial_count = len(df)
    
    # 1. Missing value analysis
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    
    # 2. Duplicate detection
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Detected {duplicates} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("Zero duplicate rows detected.")
        
    # 3. Numeric sanitize (amounts and balances must be non-negative)
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            # Clip any negative anomalies to zero
            df[col] = df[col].clip(lower=0.0)
            
    # Ensure types
    df['step'] = df['step'].astype(int)
    df['type'] = df['type'].astype(str).str.strip().str.upper()
    df['isFraud'] = df['isFraud'].astype(int)
    if 'isFlaggedFraud' in df.columns:
        df['isFlaggedFraud'] = df['isFlaggedFraud'].astype(int)
    else:
        df['isFlaggedFraud'] = 0
        
    cleaned_count = len(df)
    
    audit_report = {
        "initial_rows": initial_count,
        "cleaned_rows": cleaned_count,
        "duplicates_removed": int(duplicates),
        "total_nulls_found": int(total_nulls),
        "null_counts": null_counts,
        "fraud_count": int(df['isFraud'].sum()),
        "fraud_percentage": round(float(df['isFraud'].mean() * 100), 3)
    }
    
    logger.info(f"Data cleaning complete. Retained {cleaned_count:,} valid transactions.")
    return df, audit_report

def get_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates exploratory metrics for the transactions dataset."""
    return {
        "total_transactions": len(df),
        "total_volume": float(df['amount'].sum()),
        "average_amount": float(df['amount'].mean()),
        "median_amount": float(df['amount'].median()),
        "max_amount": float(df['amount'].max()),
        "fraud_count": int(df['isFraud'].sum()),
        "fraud_percentage": float(df['isFraud'].mean() * 100),
        "types_distribution": df['type'].value_counts().to_dict(),
        "fraud_by_type": df.groupby('type')['isFraud'].sum().to_dict(),
        "avg_amount_by_type": df.groupby('type')['amount'].mean().to_dict()
    }
