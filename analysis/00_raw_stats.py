"""
Script 0: Raw dataset statistics for merchant recommendation analysis.
Writes analysis/artifacts/raw_stats.json.
Runtime: ~3-5 min (Yelp streaming dominates).
"""
import os
import sys
import json
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    ROOT, YELP_DIR, FIN_DIR, SYN_DIR,
    SAMPLE_ROWS, save_artifact, count_lines, sample_csv,
    USER_FEATURE_NAMES, MERCHANT_FEATURE_NAMES, EDGE_FEATURE_NAMES,
)


# ------------------------------------------------------------------ #
#  Yelp                                                                #
# ------------------------------------------------------------------ #
def yelp_raw_stats() -> dict:
    print('\n=== Yelp Raw Stats ===')

    review_path   = os.path.join(YELP_DIR, 'yelp_academic_dataset_review.json')
    business_path = os.path.join(YELP_DIR, 'yelp_academic_dataset_business.json')
    user_path     = os.path.join(YELP_DIR, 'yelp_academic_dataset_user.json')
    tip_path      = os.path.join(YELP_DIR, 'yelp_academic_dataset_tip.json')

    # Line counts (fast — no load)
    n_interactions  = count_lines(review_path)
    n_merchants     = count_lines(business_path)
    n_users         = count_lines(user_path)
    n_tips          = count_lines(tip_path)
    print(f'  users={n_users:,}  merchants(businesses)={n_merchants:,}  '
          f'interactions(reviews)={n_interactions:,}  tips={n_tips:,}')

    # Stream 500K reviews for interaction value stats + temporal range
    values, dates = [], []
    fields_null   = defaultdict(int)
    fields_seen   = defaultdict(int)
    count = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= SAMPLE_ROWS:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            stars = r.get('stars')
            if stars is not None:
                values.append(float(stars))
            date_val = r.get('date', '')
            if date_val:
                dates.append(date_val)
            for key in ['review_id', 'user_id', 'business_id', 'stars', 'date']:
                fields_seen[key] += 1
                if r.get(key) is None or r.get(key) == '':
                    fields_null[key] += 1
            count += 1

    values_arr = np.array(values)
    # Yelp stars normalized to [0,1] for uniform edge weight
    norm_values = values_arr / 5.0
    interaction_value_stats = {
        'raw_mean':   float(values_arr.mean()),
        'raw_std':    float(values_arr.std()),
        'raw_min':    float(values_arr.min()),
        'raw_max':    float(values_arr.max()),
        'norm_mean':  float(norm_values.mean()),   # normalized for uniform edge weight
        'norm_std':   float(norm_values.std()),
        'norm_scale': 'stars / 5.0',
        'distribution': {str(int(s)): int((values_arr == s).sum()) for s in [1, 2, 3, 4, 5]},
    }

    valid_dates = [d for d in dates if d]
    temporal_range = {
        'min': min(valid_dates)[:10] if valid_dates else 'N/A',
        'max': max(valid_dates)[:10] if valid_dates else 'N/A',
    }

    missing_rates = {
        k: round(fields_null[k] / fields_seen[k], 4) if fields_seen[k] > 0 else 0.0
        for k in fields_seen
    }

    # Count user-user friend edges (stream user.json)
    print('  Counting social (friend) edges (streaming user.json)...')
    total_friend_edges = 0
    with open(user_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u = json.loads(line)
            except json.JSONDecodeError:
                continue
            friends_str = u.get('friends', '')
            if friends_str and friends_str != 'None':
                total_friend_edges += len([x for x in friends_str.split(',') if x.strip()])
    total_friend_edges //= 2  # undirected
    print(f'  User-user social edges: {total_friend_edges:,}')

    return {
        'dataset':                   'yelp',
        'graph_type':                'user-business bipartite (recommendation)',
        'n_users':                   n_users,
        'n_merchants':               n_merchants,
        'n_interactions':            n_interactions,
        'n_social_edges':            total_friend_edges,
        'tips_available':            n_tips,
        'temporal_range':            temporal_range,
        'interaction_value_stats':   interaction_value_stats,
        'interaction_value_meaning': 'star rating (1-5)',
        'missing_rates':             missing_rates,
        'uniform_edge_weight':       'stars / 5.0  →  [0.2, 1.0]',
        'sample_size':               count,
    }


# ------------------------------------------------------------------ #
#  Finance                                                             #
# ------------------------------------------------------------------ #
def finance_raw_stats() -> dict:
    print('\n=== Finance Raw Stats ===')

    txn_path  = os.path.join(FIN_DIR, 'transactions_data.csv')
    user_path = os.path.join(FIN_DIR, 'users_data.csv')
    card_path = os.path.join(FIN_DIR, 'cards_data.csv')

    # Small files — load fully
    users_df = pd.read_csv(user_path)
    cards_df = pd.read_csv(card_path)
    n_users  = len(users_df)
    n_cards  = len(cards_df)
    print(f'  users={n_users:,}  cards={n_cards:,}')

    user_missing = {c: round(float(users_df[c].isna().mean()), 4) for c in users_df.columns}
    card_missing = {c: round(float(cards_df[c].isna().mean()), 4) for c in cards_df.columns}

    # Chunk-read transactions
    print('  Streaming transactions_data.csv...')
    n_txn     = 0
    merchants = set()
    amounts   = []
    dates     = []
    txn_missing = defaultdict(int)
    txn_total   = defaultdict(int)

    for chunk in pd.read_csv(txn_path, chunksize=SAMPLE_ROWS, low_memory=False):
        n_txn += len(chunk)
        merchants.update(chunk['merchant_id'].dropna().astype(str).unique())
        amt = pd.to_numeric(
            chunk['amount'].astype(str).str.replace('$', '', regex=False),
            errors='coerce'
        )
        amounts.append(amt.dropna().values)
        dates.extend(chunk['date'].dropna().tolist())
        for col in chunk.columns:
            txn_total[col]   += len(chunk)
            txn_missing[col] += int(chunk[col].isna().sum())

    amounts_arr = np.concatenate(amounts)
    amounts_abs = np.abs(amounts_arr)
    # Normalize: sigmoid(log1p(|amount|) / p75) — uniform edge weight scale
    p75 = float(np.percentile(amounts_abs, 75))
    log_amt = np.log1p(amounts_abs)
    norm_values = 1.0 / (1.0 + np.exp(-log_amt / max(p75, 1e-6)))

    interaction_value_stats = {
        'raw_mean':     float(amounts_arr.mean()),
        'raw_std':      float(amounts_arr.std()),
        'raw_min':      float(amounts_arr.min()),
        'raw_max':      float(amounts_arr.max()),
        'pct_negative': float((amounts_arr < 0).mean()),
        'norm_mean':    float(norm_values.mean()),
        'norm_std':     float(norm_values.std()),
        'norm_scale':   'sigmoid(log1p(|amount|) / p75)',
    }

    dates_sorted = sorted(dates)
    temporal_range = {
        'min': str(dates_sorted[0])[:10]  if dates_sorted else 'N/A',
        'max': str(dates_sorted[-1])[:10] if dates_sorted else 'N/A',
    }

    txn_missing_rates = {
        k: round(txn_missing[k] / txn_total[k], 4) if txn_total[k] > 0 else 0.0
        for k in txn_total
    }

    n_merchants = len(merchants)
    print(f'  interactions={n_txn:,}  merchants={n_merchants:,}')

    return {
        'dataset':                   'finance',
        'graph_type':                'user-merchant bipartite (spending/recommendation)',
        'n_users':                   n_users,
        'n_merchants':               n_merchants,
        'n_interactions':            n_txn,
        'n_cards':                   n_cards,
        'temporal_range':            temporal_range,
        'interaction_value_stats':   interaction_value_stats,
        'interaction_value_meaning': 'transaction amount (USD)',
        'missing_rates': {
            'users':        user_missing,
            'cards':        card_missing,
            'transactions': txn_missing_rates,
        },
        'uniform_edge_weight':       'sigmoid(log1p(|amount|) / p75)  →  (0, 1)',
    }


# ------------------------------------------------------------------ #
#  Synthetic                                                           #
# ------------------------------------------------------------------ #
def synthetic_raw_stats() -> dict:
    print('\n=== Synthetic Raw Stats ===')

    syn_path = os.path.join(SYN_DIR, 'dataset.csv')

    n_txn     = 0
    customers = set()
    merchants = set()
    amounts   = []
    dates     = []
    missing   = defaultdict(int)
    total     = defaultdict(int)

    for chunk in pd.read_csv(syn_path, chunksize=SAMPLE_ROWS, low_memory=False):
        n_txn += len(chunk)
        customers.update(chunk['customer_id'].dropna().unique())
        merchants.update(chunk['merchant_name'].dropna().unique())
        amounts.append(chunk['amount_mnt'].dropna().values)
        dates.extend(chunk['timestamp'].dropna().tolist())
        for col in chunk.columns:
            total[col]   += len(chunk)
            missing[col] += int(chunk[col].isna().sum())

    amounts_arr = np.concatenate(amounts)
    p75 = float(np.percentile(amounts_arr.clip(min=0), 75))
    log_amt = np.log1p(amounts_arr.clip(min=0))
    norm_values = 1.0 / (1.0 + np.exp(-log_amt / max(p75, 1e-6)))

    interaction_value_stats = {
        'raw_mean':  float(amounts_arr.mean()),
        'raw_std':   float(amounts_arr.std()),
        'raw_min':   float(amounts_arr.min()),
        'raw_max':   float(amounts_arr.max()),
        'norm_mean': float(norm_values.mean()),
        'norm_std':  float(norm_values.std()),
        'norm_scale': 'sigmoid(log1p(amount) / p75)',
    }

    dates_sorted = sorted(dates)
    temporal_range = {
        'min': str(dates_sorted[0])[:10]  if dates_sorted else 'N/A',
        'max': str(dates_sorted[-1])[:10] if dates_sorted else 'N/A',
    }

    missing_rates = {
        k: round(missing[k] / total[k], 4) if total[k] > 0 else 0.0
        for k in total
    }

    n_cust  = len(customers)
    n_merch = len(merchants)
    print(f'  interactions={n_txn:,}  users(customers)={n_cust:,}  merchants={n_merch:,}')
    if n_merch < 100:
        print(f'  WARNING: only {n_merch} unique merchants — near-complete graph density expected')

    return {
        'dataset':                   'synthetic',
        'graph_type':                'user(customer)-merchant bipartite (synthetic recommendation)',
        'n_users':                   n_cust,
        'n_merchants':               n_merch,
        'n_interactions':            n_txn,
        'temporal_range':            temporal_range,
        'interaction_value_stats':   interaction_value_stats,
        'interaction_value_meaning': 'transaction amount (monetary units)',
        'missing_rates':             missing_rates,
        'uniform_edge_weight':       'sigmoid(log1p(amount) / p75)  →  (0, 1)',
        'density_note':              f'Only {n_merch} unique merchants → near-complete bipartite graph',
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print('Running raw statistics collection (merchant recommendation analysis)...')
    stats = {
        'yelp':      yelp_raw_stats(),
        'finance':   finance_raw_stats(),
        'synthetic': synthetic_raw_stats(),
        'uniform_feature_schema': {
            'user_features':     USER_FEATURE_NAMES,
            'merchant_features': MERCHANT_FEATURE_NAMES,
            'edge_features':     EDGE_FEATURE_NAMES,
            'note': (
                'All 3 datasets use identical 4D user + 4D merchant feature schema. '
                'Edge weights normalized to (0,1) via dataset-specific mapping. '
                'Enables direct model comparison across datasets.'
            ),
        },
    }
    save_artifact(stats, 'raw_stats')
    print('\nDone. Artifact: analysis/artifacts/raw_stats.json')


if __name__ == '__main__':
    main()
