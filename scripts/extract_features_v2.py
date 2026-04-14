"""
Improved 8+8 feature extraction for SelfGNN datasets.

Replaces the 4+4 uniform schema with richer features:

User (8):
  0  interaction_count        — log1p+minmax
  1  avg_interaction_value    — sigmoid(log1p/p75), already [0,1]
  2  unique_merchant_count    — log1p+minmax
  3  activity_span_days       — log1p+minmax
  4  recency_score            — exp(-days_since_last / 365), naturally [0,1]
  5  txns_per_week            — count*7/max(span_days,1), log1p+minmax
  6  value_std_norm           — std(normed_amounts), minmax
  7  repeat_visit_rate        — count/max(unique_merchants,1), log1p+minmax

Merchant (8):
  0  interaction_count        — log1p+minmax
  1  avg_interaction_value    — sigmoid(log1p/p75), already [0,1]
  2  unique_user_count        — log1p+minmax
  3  category_id              — mcc_idx, log1p+minmax
  4  txns_per_user            — count/max(unique_users,1), log1p+minmax
  5  value_std_norm           — std(normed_amounts), minmax
  6  popularity_rank          — rank by count, normalised 0-1
  7  user_repeat_rate         — fraction of users with >1 visit, minmax

avg_interaction_value stays at index 1 for both sides so the data_handler can
zero it out (--use_edge_features, no-dup mode) without touching meta.json.

Usage (run from repo root):
    python scripts/extract_features_v2.py --dataset finance-merchant
    python scripts/extract_features_v2.py --dataset synthetic-merchant
    python scripts/extract_features_v2.py --dataset yelp-merchant
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    choices=['finance-merchant', 'synthetic-merchant', 'yelp-merchant'],
                    help='Dataset name under Datasets/')
args = parser.parse_args()

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR   = os.path.join(REPO_ROOT, 'Datasets', args.dataset) + os.sep

# ---------------------------------------------------------------------------
# Load ID mappings (built by preprocess_*.ipynb)
# ---------------------------------------------------------------------------
with open(OUT_DIR + 'user2id.pkl', 'rb') as f:
    user2id = pickle.load(f)
with open(OUT_DIR + 'merchant2id.pkl', 'rb') as f:
    merchant2id = pickle.load(f)

usrnum      = len(user2id)
merchantnum = len(merchant2id)
print(f'[{args.dataset}] users={usrnum:,}  merchants={merchantnum:,}')

# ---------------------------------------------------------------------------
# Locate raw CSV
# ---------------------------------------------------------------------------
RAW_CANDIDATES = {
    'finance-merchant':   os.path.join(REPO_ROOT, 'datasetRaw', 'finance', 'transactions_data.csv'),
    'synthetic-merchant': os.path.join(REPO_ROOT, 'datasetRaw', 'synthetic', 'dataset.csv'),
    'yelp-merchant':      os.path.join(REPO_ROOT, 'datasetRaw', 'yelp', 'yelp.csv'),
}
RAW_PATH = RAW_CANDIDATES[args.dataset]
if not os.path.isfile(RAW_PATH):
    raise FileNotFoundError(f'Raw CSV not found: {RAW_PATH}')

# ---------------------------------------------------------------------------
# Dataset-specific column names
# ---------------------------------------------------------------------------
if args.dataset == 'finance-merchant':
    USER_COL   = 'client_id'
    ITEM_COL   = 'merchant_id'
    DATE_COL   = 'date'
    AMOUNT_COL = 'amount'
    MCC_COL    = 'mcc'
elif args.dataset == 'synthetic-merchant':
    USER_COL   = 'customer_id'
    ITEM_COL   = 'merchant_name'
    DATE_COL   = 'timestamp'
    AMOUNT_COL = 'amount_mnt'
    MCC_COL    = 'merchant_category_code'
else:
    # yelp — adapt if columns differ
    USER_COL   = 'user_id'
    ITEM_COL   = 'business_id'
    DATE_COL   = 'date'
    AMOUNT_COL = 'stars'
    MCC_COL    = None

USE_COLS = [USER_COL, ITEM_COL, DATE_COL, AMOUNT_COL]
if MCC_COL:
    USE_COLS.append(MCC_COL)

# ---------------------------------------------------------------------------
# Pass 1: compute p75 of log|amount| for normalisation
# ---------------------------------------------------------------------------
print('Pass 1: computing p75 ...')
all_amounts = []
for chunk in pd.read_csv(RAW_PATH, chunksize=500_000, low_memory=False, usecols=USE_COLS):
    chunk[USER_COL] = chunk[USER_COL].astype(str)
    chunk[ITEM_COL] = chunk[ITEM_COL].astype(str)
    if args.dataset == 'finance-merchant':
        chunk[AMOUNT_COL] = pd.to_numeric(
            chunk[AMOUNT_COL].astype(str).str.replace('$', '', regex=False), errors='coerce')
    else:
        chunk[AMOUNT_COL] = pd.to_numeric(chunk[AMOUNT_COL], errors='coerce')
    chunk = chunk[
        chunk[USER_COL].isin(user2id) &
        chunk[ITEM_COL].isin(merchant2id)
    ].dropna(subset=[AMOUNT_COL])
    all_amounts.append(chunk[AMOUNT_COL].abs().clip(lower=0).values)

amounts_flat = np.concatenate(all_amounts)
log_amounts  = np.log1p(amounts_flat)
p75          = float(max(np.percentile(log_amounts, 75), 1.0))
print(f'  p75(log|amount|) = {p75:.4f}')

def norm_amount(amt: np.ndarray) -> np.ndarray:
    """sigmoid(log1p(|amt|) / p75) → (0,1)"""
    return 1.0 / (1.0 + np.exp(-np.log1p(np.clip(amt, 0, None)) / p75))

# ---------------------------------------------------------------------------
# Pass 2: accumulate per-user / per-merchant statistics
# ---------------------------------------------------------------------------
print('Pass 2: accumulating statistics ...')

# User accumulators
u_count     = np.zeros(usrnum, dtype=np.int64)
u_val_sum   = np.zeros(usrnum, dtype=np.float64)
u_val_sq    = np.zeros(usrnum, dtype=np.float64)   # for std
u_merchants = [set() for _ in range(usrnum)]
u_min_ts    = np.full(usrnum,  np.inf)
u_max_ts    = np.full(usrnum, -np.inf)

# Merchant accumulators
m_count       = np.zeros(merchantnum, dtype=np.int64)
m_val_sum     = np.zeros(merchantnum, dtype=np.float64)
m_val_sq      = np.zeros(merchantnum, dtype=np.float64)  # for std
m_users       = [set() for _ in range(merchantnum)]
m_mcc: dict   = {}

# Edge accumulators (for user_repeat_rate)
edge_cnt: dict = defaultdict(int)

global_max_ts = -np.inf

for chunk in pd.read_csv(RAW_PATH, chunksize=500_000, low_memory=False, usecols=USE_COLS):
    chunk[USER_COL] = chunk[USER_COL].astype(str)
    chunk[ITEM_COL] = chunk[ITEM_COL].astype(str)
    if args.dataset == 'finance-merchant':
        chunk[AMOUNT_COL] = pd.to_numeric(
            chunk[AMOUNT_COL].astype(str).str.replace('$', '', regex=False), errors='coerce')
    else:
        chunk[AMOUNT_COL] = pd.to_numeric(chunk[AMOUNT_COL], errors='coerce')
    chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors='coerce')
    chunk = chunk[
        chunk[USER_COL].isin(user2id) &
        chunk[ITEM_COL].isin(merchant2id)
    ].dropna(subset=[AMOUNT_COL, DATE_COL])

    chunk['uid']      = chunk[USER_COL].map(user2id)
    chunk['mid']      = chunk[ITEM_COL].map(merchant2id)
    chunk['norm_val'] = norm_amount(chunk[AMOUNT_COL].abs().values)
    chunk['unix_ts']  = chunk[DATE_COL].astype(np.int64) // 10**9

    iter_cols = ['uid', 'mid', 'norm_val', 'unix_ts']
    if MCC_COL and MCC_COL in chunk.columns:
        iter_cols.append(MCC_COL)
        has_mcc = True
    else:
        has_mcc = False

    for row in chunk[iter_cols].itertuples(index=False):
        uid, mid = int(row.uid), int(row.mid)
        v  = float(row.norm_val)
        ts = int(row.unix_ts)

        u_count[uid]  += 1
        u_val_sum[uid] += v
        u_val_sq[uid]  += v * v
        u_merchants[uid].add(mid)
        if ts < u_min_ts[uid]: u_min_ts[uid] = ts
        if ts > u_max_ts[uid]: u_max_ts[uid] = ts
        if ts > global_max_ts: global_max_ts = ts

        m_count[mid]  += 1
        m_val_sum[mid] += v
        m_val_sq[mid]  += v * v
        m_users[mid].add(uid)

        if has_mcc and mid not in m_mcc:
            mcc_val = getattr(row, MCC_COL.replace('-', '_'), None)
            if mcc_val is not None and pd.notna(mcc_val):
                m_mcc[mid] = int(mcc_val)

        edge_cnt[(uid, mid)] += 1

print(f'  Total events processed: {u_count.sum():,}')

# ---------------------------------------------------------------------------
# Compute derived statistics
# ---------------------------------------------------------------------------
SECS_PER_DAY  = 86400.0
SECS_PER_YEAR = 365.0 * SECS_PER_DAY

# -- User derived --
u_avg_val   = np.where(u_count > 0, u_val_sum / u_count, 0.0).astype(np.float32)
u_var       = np.maximum(u_val_sq / np.maximum(u_count, 1) - (u_val_sum / np.maximum(u_count, 1))**2, 0.0)
u_val_std   = np.sqrt(u_var).astype(np.float32)
u_unique_m  = np.array([len(s) for s in u_merchants], dtype=np.float32)
u_span_days = np.where(
    np.isfinite(u_min_ts) & np.isfinite(u_max_ts),
    (u_max_ts - u_min_ts) / SECS_PER_DAY, 0.0,
).astype(np.float32)
u_recency   = np.where(
    np.isfinite(u_max_ts),
    np.exp(-np.maximum(0.0, global_max_ts - u_max_ts) / SECS_PER_YEAR),
    0.0,
).astype(np.float32)
u_txns_per_week = (u_count.astype(np.float32) * 7.0
                   / np.maximum(u_span_days, 1.0))
u_repeat_rate   = (u_count.astype(np.float32)
                   / np.maximum(u_unique_m, 1.0))

# -- Merchant derived --
m_avg_val     = np.where(m_count > 0, m_val_sum / m_count, 0.0).astype(np.float32)
m_var         = np.maximum(m_val_sq / np.maximum(m_count, 1) - (m_val_sum / np.maximum(m_count, 1))**2, 0.0)
m_val_std     = np.sqrt(m_var).astype(np.float32)
m_unique_u    = np.array([len(s) for s in m_users], dtype=np.float32)
m_txns_per_u  = (m_count.astype(np.float32)
                 / np.maximum(m_unique_u, 1.0))

# user_repeat_rate: fraction of (uid,mid) pairs where user visited merchant >1 time
m_repeat_u = np.zeros(merchantnum, dtype=np.float32)
for (uid, mid), cnt in edge_cnt.items():
    if cnt > 1:
        m_repeat_u[mid] += 1.0
m_user_repeat_rate = m_repeat_u / np.maximum(m_unique_u, 1.0)

# popularity_rank: 0 = least popular, 1 = most popular
m_count_order = np.argsort(m_count)
m_pop_rank    = np.zeros(merchantnum, dtype=np.float32)
m_pop_rank[m_count_order] = np.linspace(0.0, 1.0, merchantnum)

# MCC index
unique_mccs = sorted(set(m_mcc.values())) if m_mcc else [0]
mcc2idx     = {v: i for i, v in enumerate(unique_mccs)}
m_cat       = np.array(
    [float(mcc2idx.get(m_mcc.get(mid, 0), 0)) for mid in range(merchantnum)],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Assemble feature matrices (raw, pre-normalisation)
# ---------------------------------------------------------------------------
user_features = np.stack([
    u_count.astype(np.float32),   # 0: interaction_count
    u_avg_val,                     # 1: avg_interaction_value
    u_unique_m,                    # 2: unique_merchant_count
    u_span_days,                   # 3: activity_span_days
    u_recency,                     # 4: recency_score  (already [0,1])
    u_txns_per_week,               # 5: txns_per_week
    u_val_std,                     # 6: value_std_norm
    u_repeat_rate,                 # 7: repeat_visit_rate
], axis=1)

merchant_features = np.stack([
    m_count.astype(np.float32),   # 0: interaction_count
    m_avg_val,                     # 1: avg_interaction_value
    m_unique_u,                    # 2: unique_user_count
    m_cat,                         # 3: category_id (MCC index)
    m_txns_per_u,                  # 4: txns_per_user
    m_val_std,                     # 5: value_std_norm
    m_pop_rank,                    # 6: popularity_rank  (already [0,1])
    m_user_repeat_rate,            # 7: user_repeat_rate (already [0,1])
], axis=1)

# ---------------------------------------------------------------------------
# Normalise
# Columns already in [0,1]: 1 (avg_value), 4 (recency), 6 (pop_rank), 7 (repeat_rate)
# Remaining columns: log1p+minmax
# ---------------------------------------------------------------------------
SKIP_NORM_USER  = {1, 4}           # already [0,1]
SKIP_NORM_MERCH = {1, 6, 7}        # already [0,1]

def log_minmax(arr: np.ndarray) -> np.ndarray:
    v = np.log1p(np.clip(arr, 0, None))
    mn, mx = v.min(), v.max()
    return ((v - mn) / max(mx - mn, 1e-8)).astype(np.float32)

def minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / max(mx - mn, 1e-8)).astype(np.float32)

user_norm = user_features.copy()
for col in range(user_features.shape[1]):
    if col in SKIP_NORM_USER:
        continue
    if col == 6:   # value_std: already in [0,1] range, just minmax
        user_norm[:, col] = minmax(user_features[:, col])
    else:
        user_norm[:, col] = log_minmax(user_features[:, col])

merch_norm = merchant_features.copy()
for col in range(merchant_features.shape[1]):
    if col in SKIP_NORM_MERCH:
        continue
    if col == 5:   # value_std: minmax
        merch_norm[:, col] = minmax(merchant_features[:, col])
    else:
        merch_norm[:, col] = log_minmax(merchant_features[:, col])

print(f'User   features: {user_norm.shape}  '
      f'min/max per col: {user_norm.min(0).round(3)} / {user_norm.max(0).round(3)}')
print(f'Merch  features: {merch_norm.shape}  '
      f'min/max per col: {merch_norm.min(0).round(3)} / {merch_norm.max(0).round(3)}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.save(OUT_DIR + 'user_features.npy',     user_norm)
np.save(OUT_DIR + 'merchant_features.npy', merch_norm)

meta = {
    'user_feature_names': [
        'interaction_count', 'avg_interaction_value', 'unique_merchant_count',
        'activity_span_days', 'recency_score', 'txns_per_week',
        'value_std_norm', 'repeat_visit_rate',
    ],
    'merchant_feature_names': [
        'interaction_count', 'avg_interaction_value', 'unique_user_count',
        'category_id', 'txns_per_user', 'value_std_norm',
        'popularity_rank', 'user_repeat_rate',
    ],
    'edge_feature_names':  ['normalized_interaction_value'],
    'user_value_col':      1,
    'merchant_value_col':  1,
    'edge_normalization':  'sigmoid(log1p(|amount|) / p75)',
    'dataset':             args.dataset,
    'schema':              '8+8',
    'mcc_unique_count':    len(unique_mccs),
    'p75':                 p75,
}
with open(OUT_DIR + 'feature_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nSaved:')
for fname in ['user_features.npy', 'merchant_features.npy', 'feature_meta.json']:
    path = OUT_DIR + fname
    sz   = os.path.getsize(path) / 1024
    print(f'  {fname:<35} {sz:.0f} KB')
print('Done.')
