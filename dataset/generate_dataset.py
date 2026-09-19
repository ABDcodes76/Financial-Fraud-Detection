import os
import argparse
import numpy as np
import pandas as pd

def generate_paysim_dataset(num_rows=60000, random_state=42):
    """
    Generates a high-fidelity synthetic financial transaction dataset modeled
    strictly after the PaySim benchmark (Lopez-Rojas et al.).
    
    Columns:
      step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
      nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    """
    print(f"[PaySim Generator] Generating {num_rows} realistic financial transactions...")
    np.random.seed(random_state)
    
    # 1. Step (1 to 744 hours, representing 31 days)
    # Peak activity during daytime (hours 9-20), lower at night
    days = np.random.choice(np.arange(1, 32), size=num_rows)
    hour_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 00-05
        0.03, 0.05, 0.06, 0.07, 0.07, 0.07,  # 06-11
        0.08, 0.08, 0.08, 0.07, 0.07, 0.06,  # 12-17
        0.05, 0.04, 0.03, 0.02, 0.01, 0.01   # 18-23
    ])
    hour_probs = hour_probs / hour_probs.sum()
    hours = np.random.choice(np.arange(24), size=num_rows, p=hour_probs)
    steps = (days - 1) * 24 + hours + 1
    steps = np.clip(steps, 1, 744)
    
    # 2. Transaction Types distribution matching PaySim
    # PAYMENT ~35%, CASH_OUT ~34%, CASH_IN ~22%, TRANSFER ~8%, DEBIT ~1%
    type_choices = ['PAYMENT', 'CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT']
    type_probs = [0.35, 0.34, 0.22, 0.08, 0.01]
    types = np.random.choice(type_choices, size=num_rows, p=type_probs)
    
    # Initial balance generators
    oldbalance_orig = np.zeros(num_rows, dtype=np.float64)
    newbalance_orig = np.zeros(num_rows, dtype=np.float64)
    oldbalance_dest = np.zeros(num_rows, dtype=np.float64)
    newbalance_dest = np.zeros(num_rows, dtype=np.float64)
    amounts = np.zeros(num_rows, dtype=np.float64)
    is_fraud = np.zeros(num_rows, dtype=np.int32)
    is_flagged_fraud = np.zeros(num_rows, dtype=np.int32)
    
    name_orig = [f"C{np.random.randint(100000000, 999999999)}" for _ in range(num_rows)]
    name_dest = []
    
    # Target fraud count (~1.0% of total, concentrated exclusively in TRANSFER and CASH_OUT)
    transfer_cashout_indices = [i for i, t in enumerate(types) if t in ('TRANSFER', 'CASH_OUT')]
    target_fraud_count = int(num_rows * 0.0105)  # ~630 fraud cases
    fraud_indices = set(np.random.choice(transfer_cashout_indices, size=target_fraud_count, replace=False))
    
    for i in range(num_rows):
        ttype = types[i]
        is_fr = 1 if i in fraud_indices else 0
        is_fraud[i] = is_fr
        
        if is_fr:
            # FRAUDULENT TRANSACTIONS:
            # Draining victim accounts: amount is often the full oldbalanceOrg
            if np.random.rand() < 0.75:
                old_orig = np.round(np.random.exponential(scale=350000) + 15000, 2)
                amt = old_orig  # Drain account completely
                new_orig = 0.0
            else:
                old_orig = np.round(np.random.exponential(scale=500000) + 50000, 2)
                amt = np.round(np.random.uniform(0.7, 0.99) * old_orig, 2)
                new_orig = np.round(max(0.0, old_orig - amt), 2)
                
            oldbalance_orig[i] = old_orig
            amounts[i] = amt
            newbalance_orig[i] = new_orig
            
            # Destination account
            dest_id = f"C{np.random.randint(100000000, 999999999)}"
            name_dest.append(dest_id)
            
            # Fraudulent destinations often start with 0 balance
            old_dest = 0.0 if np.random.rand() < 0.8 else np.round(np.random.exponential(scale=50000), 2)
            oldbalance_dest[i] = old_dest
            
            # Often cash-out accomplices drain immediately, so newbalanceDest is 0 or amt
            if ttype == 'CASH_OUT':
                newbalance_dest[i] = np.round(old_dest + amt, 2) if np.random.rand() < 0.4 else 0.0
            else:
                newbalance_dest[i] = np.round(old_dest + amt, 2)
                
            # Flagged fraud rule: single TRANSFER > 200,000
            if ttype == 'TRANSFER' and amt > 200000 and np.random.rand() < 0.45:
                is_flagged_fraud[i] = 1
                
        else:
            # LEGITIMATE TRANSACTIONS
            if ttype == 'PAYMENT':
                # Customer to Merchant
                dest_id = f"M{np.random.randint(100000000, 999999999)}"
                name_dest.append(dest_id)
                # Payments are usually smaller amounts
                amt = np.round(np.random.lognormal(mean=7.0, sigma=1.2), 2) + 1.0
                amt = min(amt, 150000.0)
                old_orig = np.round(amt + np.random.exponential(scale=20000), 2)
                new_orig = np.round(max(0.0, old_orig - amt), 2)
                # In PaySim, merchant balances are not tracked (0.0)
                old_dest = 0.0
                new_dest = 0.0
                
            elif ttype == 'CASH_IN':
                # Depositing funds
                dest_id = f"C{np.random.randint(100000000, 999999999)}"
                name_dest.append(dest_id)
                amt = np.round(np.random.lognormal(mean=10.5, sigma=1.1), 2) + 10.0
                old_orig = np.round(np.random.exponential(scale=100000), 2)
                new_orig = np.round(old_orig + amt, 2)
                old_dest = np.round(np.random.exponential(scale=100000), 2)
                new_dest = np.round(max(0.0, old_dest - amt), 2)
                
            elif ttype == 'CASH_OUT':
                # Legitimate cash withdrawal
                dest_id = f"C{np.random.randint(100000000, 999999999)}"
                name_dest.append(dest_id)
                amt = np.round(np.random.lognormal(mean=10.8, sigma=1.0), 2) + 20.0
                old_orig = np.round(amt + np.random.exponential(scale=50000), 2)
                new_orig = np.round(max(0.0, old_orig - amt), 2)
                old_dest = np.round(np.random.exponential(scale=150000), 2)
                new_dest = np.round(old_dest + amt, 2)
                
            elif ttype == 'TRANSFER':
                # Legitimate fund transfer between customers
                dest_id = f"C{np.random.randint(100000000, 999999999)}"
                name_dest.append(dest_id)
                amt = np.round(np.random.lognormal(mean=11.2, sigma=1.1), 2) + 50.0
                old_orig = np.round(amt + np.random.exponential(scale=80000), 2)
                new_orig = np.round(max(0.0, old_orig - amt), 2)
                old_dest = np.round(np.random.exponential(scale=100000), 2)
                new_dest = np.round(old_dest + amt, 2)
                
            else:  # DEBIT
                dest_id = f"C{np.random.randint(100000000, 999999999)}"
                name_dest.append(dest_id)
                amt = np.round(np.random.lognormal(mean=7.5, sigma=0.8), 2) + 5.0
                old_orig = np.round(amt + np.random.exponential(scale=15000), 2)
                new_orig = np.round(max(0.0, old_orig - amt), 2)
                old_dest = np.round(np.random.exponential(scale=20000), 2)
                new_dest = np.round(old_dest + amt, 2)
                
            amounts[i] = amt
            oldbalance_orig[i] = old_orig
            newbalance_orig[i] = new_orig
            oldbalance_dest[i] = old_dest
            newbalance_dest[i] = new_dest

    df = pd.DataFrame({
        'step': steps,
        'type': types,
        'amount': amounts,
        'nameOrig': name_orig,
        'oldbalanceOrg': oldbalance_orig,
        'newbalanceOrig': newbalance_orig,
        'nameDest': name_dest,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest,
        'isFraud': is_fraud,
        'isFlaggedFraud': is_flagged_fraud
    })
    
    # Introduce small realistic noise/anomalies (balance discrepancies)
    noise_idx = np.random.choice(num_rows, size=int(num_rows * 0.03), replace=False)
    for idx in noise_idx:
        if df.loc[idx, 'type'] in ['CASH_OUT', 'TRANSFER'] and df.loc[idx, 'isFraud'] == 0:
            df.loc[idx, 'newbalanceOrig'] = max(0.0, df.loc[idx, 'newbalanceOrig'] + np.random.uniform(-500, 500))
            
    print(f"[PaySim Generator] Successfully created dataset with shape {df.shape}")
    print(f"[PaySim Generator] Fraud count: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    print(f"[PaySim Generator] Types breakdown:\n{df['type'].value_counts(normalize=True).round(3)}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PaySim Synthetic Financial Dataset")
    parser.add_argument('--rows', type=int, default=60000, help="Number of rows to generate")
    parser.add_argument('--output', type=str, default="dataset/transactions.csv", help="Output CSV path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_paysim_dataset(num_rows=args.rows)
    df.to_csv(args.output, index=False)
    print(f"[PaySim Generator] Saved dataset to {args.output}")
