import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from src.utils import logger, save_model, load_model

TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

# Core feature names that will be produced by the pipeline
FEATURE_NAMES = [
    'step',
    'amount',
    'log_amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'orig_balance_diff',
    'dest_balance_diff',
    'error_balance_orig',
    'error_balance_dest',
    'amount_to_oldbalance_ratio',
    'is_drained_orig',
    'orig_zero_balance_transfer',
    'is_merchant_dest',
    'hour_of_day',
    'day_of_month',
    'is_night',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER'
]

def engineer_features(
    df: pd.DataFrame,
    is_training: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], StandardScaler]:
    """
    Computes domain-specific financial features for both training and inference.
    Computes record-level domain indicators.
    """
    logger.info("Computing domain-engineered financial features...")
    data = df.copy()
    
    # 1. Log Amount
    data['log_amount'] = np.log1p(data['amount'])
    
    # 2. Balance Differences
    data['orig_balance_diff'] = data['oldbalanceOrg'] - data['newbalanceOrig']
    data['dest_balance_diff'] = data['newbalanceDest'] - data['oldbalanceDest']
    
    # 3. Discrepancy / Balance Errors
    # Origin: For legitimate outflows, new = old - amount => new + amount - old should equal 0
    data['error_balance_orig'] = data['newbalanceOrig'] + data['amount'] - data['oldbalanceOrg']
    
    # Destination: For legitimate inflows, new = old + amount => old + amount - new should equal 0
    data['error_balance_dest'] = data['oldbalanceDest'] + data['amount'] - data['newbalanceDest']
    
    # 4. Ratios and behavioural flags
    data['amount_to_oldbalance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1.0)
    data['is_drained_orig'] = ((data['oldbalanceOrg'] > 0) & (data['newbalanceOrig'] == 0.0)).astype(int)
    data['orig_zero_balance_transfer'] = ((data['oldbalanceOrg'] == 0.0) & (data['amount'] > 0)).astype(int)
    
    # Check if recipient is a merchant
    if 'nameDest' in data.columns:
        data['is_merchant_dest'] = data['nameDest'].astype(str).str.startswith('M').astype(int)
    else:
        data['is_merchant_dest'] = 0
        
    # 5. Temporal Features
    data['hour_of_day'] = (data['step'] % 24).astype(int)
    data['day_of_month'] = ((data['step'] // 24) % 30 + 1).astype(int)
    data['is_night'] = ((data['hour_of_day'] >= 0) & (data['hour_of_day'] <= 5)).astype(int)
    
    # 6. One-hot encode transaction type
    for t in TRANSACTION_TYPES:
        data[f'type_{t}'] = (data['type'] == t).astype(int)
        
    # Select feature matrix
    X = data[FEATURE_NAMES].copy()
    
    # Handle scaling
    continuous_features = [
        'step', 'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'orig_balance_diff',
        'dest_balance_diff', 'error_balance_orig', 'error_balance_dest',
        'amount_to_oldbalance_ratio', 'hour_of_day', 'day_of_month'
    ]
    
    if is_training:
        scaler = StandardScaler()
        scaler.fit(X[continuous_features])
        save_model({"scaler": scaler, "features": FEATURE_NAMES, "continuous": continuous_features}, "preprocessing.pkl")
    else:
        if scaler is None:
            preproc = load_model("preprocessing.pkl")
            scaler = preproc["scaler"]
            
    # Target label extraction
    y = data['isFraud'] if 'isFraud' in data.columns else None
    
    logger.info(f"Feature engineering complete. Produced matrix of shape {X.shape}.")
    return X, y, FEATURE_NAMES, scaler

def prepare_single_transaction(tx_dict: Dict[str, Any], scaler: StandardScaler) -> pd.DataFrame:
    """
    Transforms a single raw transaction dictionary into the exact feature format
    expected by the trained models.
    """
    row = {
        'step': int(tx_dict.get('step', 1)),
        'type': str(tx_dict.get('type', 'TRANSFER')).upper(),
        'amount': float(tx_dict.get('amount', 0.0)),
        'oldbalanceOrg': float(tx_dict.get('oldbalanceOrg', 0.0)),
        'newbalanceOrig': float(tx_dict.get('newbalanceOrig', 0.0)),
        'oldbalanceDest': float(tx_dict.get('oldbalanceDest', 0.0)),
        'newbalanceDest': float(tx_dict.get('newbalanceDest', 0.0)),
        'nameOrig': str(tx_dict.get('nameOrig', 'C100000000')),
        'nameDest': str(tx_dict.get('nameDest', 'C200000000')),
    }
    
    df_single = pd.DataFrame([row])
    X_single, _, _, _ = engineer_features(df_single, is_training=False, scaler=scaler)
    return X_single
