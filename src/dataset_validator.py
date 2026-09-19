import os
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from src.utils import logger

REQUIRED_COLUMNS = [
    'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest'
]

OPTIONAL_COLUMNS = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']

ALLOWED_TRANSACTION_TYPES = {'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}

@dataclass
class ValidationResult:
    is_valid: bool
    row_count: int = 0
    col_count: int = 0
    missing_required: List[str] = field(default_factory=list)
    present_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    has_ground_truth: bool = False
    duplicate_count: int = 0
    null_counts: Dict[str, int] = field(default_factory=dict)
    invalid_types: List[str] = field(default_factory=list)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None


def validate_dataset(input_source: Union[pd.DataFrame, io.BytesIO, io.StringIO, str, bytes, Any]) -> ValidationResult:
    """
    Validates an uploaded or in-memory financial dataset against the canonical FINSEC AI schema.
    Performs type coercion, null checks, duplicate analysis, and sanity sanitization.
    """
    errors = []
    warnings = []
    
    # 1. Load into DataFrame
    df = None
    try:
        if isinstance(input_source, pd.DataFrame):
            df = input_source.copy()
        elif isinstance(input_source, (io.BytesIO, io.StringIO)):
            input_source.seek(0)
            df = pd.read_csv(input_source)
        elif isinstance(input_source, bytes):
            df = pd.read_csv(io.BytesIO(input_source))
        elif isinstance(input_source, str):
            if os.path.exists(input_source):
                df = pd.read_csv(input_source)
            else:
                df = pd.read_csv(io.StringIO(input_source))
        elif hasattr(input_source, "read"):
            input_source.seek(0)
            df = pd.read_csv(input_source)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_messages=[f"Failed to parse CSV file: {str(e)}"]
        )

    if df is None or len(df) == 0:
        return ValidationResult(
            is_valid=False,
            error_messages=["Uploaded file is empty or contains no records."]
        )

    row_count, col_count = df.shape
    present_cols = list(df.columns)
    
    # 2. Check required schema columns
    missing_required = [col for col in REQUIRED_COLUMNS if col not in present_cols]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")

    extra_cols = [col for col in present_cols if col not in REQUIRED_COLUMNS and col not in OPTIONAL_COLUMNS]
    if extra_cols:
        warnings.append(f"Unused non-standard columns detected: {', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}")

    has_ground_truth = 'isFraud' in present_cols

    if errors:
        return ValidationResult(
            is_valid=False,
            row_count=row_count,
            col_count=col_count,
            missing_required=missing_required,
            present_columns=present_cols,
            extra_columns=extra_cols,
            has_ground_truth=has_ground_truth,
            error_messages=errors,
            warning_messages=warnings
        )

    # 3. Clean and sanitize copy
    clean_df = df.copy()
    
    # Duplicate analysis
    duplicate_count = int(clean_df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count:,} duplicate rows detected. Automatic deduplication applied.")
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    # Null analysis
    null_counts = {str(k): int(v) for k, v in clean_df[REQUIRED_COLUMNS].isnull().sum().items() if v > 0}
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        warnings.append(f"{total_nulls:,} missing/null values imputed with standard financial defaults.")

    # Numeric coercions & clipping negatives
    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
        clean_df[col] = clean_df[col].clip(lower=0.0)

    clean_df['step'] = clean_df['step'].astype(int)

    # Transaction type validation
    clean_df['type'] = clean_df['type'].astype(str).str.strip().str.upper()
    found_types = set(clean_df['type'].unique())
    invalid_types = list(found_types - ALLOWED_TRANSACTION_TYPES)
    if invalid_types:
        errors.append(f"Invalid transaction types found: {', '.join(invalid_types)}. Allowed types: {', '.join(sorted(ALLOWED_TRANSACTION_TYPES))}")

    type_dist = clean_df['type'].value_counts().to_dict()

    # Optional string columns
    if 'nameOrig' not in clean_df.columns:
        clean_df['nameOrig'] = [f"C{1000000 + i}" for i in range(len(clean_df))]
    else:
        clean_df['nameOrig'] = clean_df['nameOrig'].fillna("UNKNOWN_ORIG").astype(str)

    if 'nameDest' not in clean_df.columns:
        clean_df['nameDest'] = [f"M{2000000 + i}" for i in range(len(clean_df))]
    else:
        clean_df['nameDest'] = clean_df['nameDest'].fillna("UNKNOWN_DEST").astype(str)

    # Optional Ground Truth
    if has_ground_truth:
        clean_df['isFraud'] = pd.to_numeric(clean_df['isFraud'], errors='coerce').fillna(0).astype(int)
        clean_df['isFraud'] = clean_df['isFraud'].clip(lower=0, upper=1)
        
    if 'isFlaggedFraud' in clean_df.columns:
        clean_df['isFlaggedFraud'] = pd.to_numeric(clean_df['isFlaggedFraud'], errors='coerce').fillna(0).astype(int)
    else:
        clean_df['isFlaggedFraud'] = 0

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        row_count=row_count,
        col_count=col_count,
        missing_required=missing_required,
        present_columns=present_cols,
        extra_columns=extra_cols,
        has_ground_truth=has_ground_truth,
        duplicate_count=duplicate_count,
        null_counts=null_counts,
        invalid_types=invalid_types,
        type_distribution=type_dist,
        error_messages=errors,
        warning_messages=warnings,
        cleaned_df=clean_df if is_valid else None
    )


def generate_sample_template() -> pd.DataFrame:
    """
    Generates a realistic 10-row dataset covering various payment types,
    legitimate behavior, account drains, high-value transfers, and merchant payments.
    """
    sample_data = [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "nameDest": "M1979787155",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        },
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 1864.28,
            "nameOrig": "C1666544295",
            "oldbalanceOrg": 21249.0,
            "newbalanceOrig": 19384.72,
            "nameDest": "M2044282225",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        },
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0
        },
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 181.0,
            "nameOrig": "C840083671",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C38997010",
            "oldbalanceDest": 21182.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0
        },
        {
            "step": 2,
            "type": "PAYMENT",
            "amount": 11668.14,
            "nameOrig": "C2048537720",
            "oldbalanceOrg": 41554.0,
            "newbalanceOrig": 29885.86,
            "nameDest": "M1230701703",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        },
        {
            "step": 2,
            "type": "TRANSFER",
            "amount": 2806.0,
            "nameOrig": "C1420196421",
            "oldbalanceOrg": 2806.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C972765878",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0
        },
        {
            "step": 3,
            "type": "CASH_IN",
            "amount": 15800.0,
            "nameOrig": "C1900381023",
            "oldbalanceOrg": 54000.0,
            "newbalanceOrig": 69800.0,
            "nameDest": "C482910481",
            "oldbalanceDest": 120000.0,
            "newbalanceDest": 104200.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        },
        {
            "step": 4,
            "type": "DEBIT",
            "amount": 4200.50,
            "nameOrig": "C1192837465",
            "oldbalanceOrg": 85000.0,
            "newbalanceOrig": 80799.50,
            "nameDest": "C992817264",
            "oldbalanceDest": 45000.0,
            "newbalanceDest": 49200.50,
            "isFraud": 0,
            "isFlaggedFraud": 0
        },
        {
            "step": 5,
            "type": "TRANSFER",
            "amount": 450000.0,
            "nameOrig": "C1102938475",
            "oldbalanceOrg": 450000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C883719203",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 450000.0,
            "isFraud": 1,
            "isFlaggedFraud": 0
        },
        {
            "step": 6,
            "type": "CASH_OUT",
            "amount": 32000.0,
            "nameOrig": "C998877665",
            "oldbalanceOrg": 55000.0,
            "newbalanceOrig": 23000.0,
            "nameDest": "C112233445",
            "oldbalanceDest": 150000.0,
            "newbalanceDest": 182000.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        }
    ]
    return pd.DataFrame(sample_data)
