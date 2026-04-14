"""
Improved 8+8 feature extraction for SelfGNN datasets.

Replaces the 4+4 uniform schema with richer features:

User (8):
  0  interaction_count        — log1p+minmax
  1  avg_interaction_value    — already [0,1] (sigmoid/stars-norm)
  2  unique_merchant_count    — log1p+minmax
  3  activity_span_days       — log1p+minmax
  4  recency_score            — exp(-days_since_last / 365), naturally [0,1]
  5  txns_per_week            — count*7/max(span_days,1), log1p+minmax
  6  value_std_norm           — std(normed_values), minmax
  7  repeat_visit_rate        — count/max(unique_merchants,1), log1p+minmax

Merchant (8):
  0  interaction_count        — log1p+minmax
  1  avg_interaction_value    — already [0,1]
  2  unique_user_count        — log1p+minmax
  3  category_id              — category idx, log1p+minmax
  4  txns_per_user            — count/max(unique_users,1), log1p+minmax
  5  value_std_norm           — std(normed_values), minmax
  6  popularity_rank          — rank by count, normalised 0-1
  7  user_repeat_rate         — fraction of users with >1 visit, naturally [0,1]

avg_interaction_value stays at index 1 for both sides so the data_handler can
zero it out (--use_edge_features, no-dup mode) without touching meta.json.

Dataset-specific raw sources:
  finance-merchant  : datasetRaw/finance/transactions_data.csv  (CSV, amount col)
  synthetic-merchant: datasetRaw/synthetic/dataset.csv          (CSV, amount_mnt col)
  yelp-merchant     : datasetRaw/yelp/yelp_academic_dataset_review.json +
                      yelp_academic_dataset_business.json       (JSONL, stars col)

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
from datetime import datetime

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


# ===========================================================================
# YELP — JSONL path (completely separate from CSV path)
# ===========================================================================
if args.dataset == 'yelp-merchant':
    RAW_DIR      = os.path.join(REPO_ROOT, 'datasetRaw', 'yelp') + os.sep
    REVIEW_PATH  = RAW_DIR + 'yelp_academic_dataset_review.json'
    BUSINESS_PATH = RAW_DIR + 'yelp_academic_dataset_business.json'

    # --- Category mapping from business.json ---
    print('Loading business categories ...')
    bid_to_cat: dict = {}
    with open(BUSINESS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            b   = json.loads(line.strip())
            bid = b.get('business_id', '')
            if bid not in merchant2id:
                continue
            cats = b.get('categories') or ''
            toks = [c.strip() for c in cats.split(',') if c.strip()]
            bid_to_cat[bid] = toks[0] if toks else 'Unknown'

    all_cats = sorted(set(bid_to_cat.values()))
    cat2id   = {c: i for i, c in enumerate(all_cats)}
    merchant_cat = np.zeros(merchantnum, dtype=np.float32)
    for bid, cat in bid_to_cat.items():
        mid = merchant2id.get(bid)
        if mid is not None:
            merchant_cat[mid] = float(cat2id[cat])
    print(f'  {len(all_cats):,} unique categories')

    def parse_ts_yelp(s: str) -> float:
        try:
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception:
            return float('nan')

    # --- Accumulators ---
    u_count     = np.zeros(usrnum,      dtype=np.int64)
    u_val_sum   = np.zeros(usrnum,      dtype=np.float64)
    u_val_sq    = np.zeros(usrnum,      dtype=np.float64)
    u_merchants = [set() for _ in range(usrnum)]
    u_min_ts    = np.full(usrnum,  np.inf)
    u_max_ts    = np.full(usrnum, -np.inf)

    m_count     = np.zeros(merchantnum, dtype=np.int64)
    m_val_sum   = np.zeros(merchantnum, dtype=np.float64)
    m_val_sq    = np.zeros(merchantnum, dtype=np.float64)
    m_users     = [set() for _ in range(merchantnum)]

    edge_cnt: dict = defaultdict(int)
    global_max_ts  = -np.inf

    print('Streaming reviews ...')
    n_kept = 0
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            rv      = json.loads(line.strip())
            uid_str = rv.get('user_id', '')
            bid     = rv.get('business_id', '')
            stars   = rv.get('stars')
            if uid_str not in user2id or bid not in merchant2id or stars is None:
                continue
            uid  = user2id[uid_str]
            mid  = merchant2id[bid]
            v    = float(stars) / 5.0   # normalise to [0.2, 1.0]

            u_count[uid]   += 1
            u_val_sum[uid]  += v
            u_val_sq[uid]   += v * v
            u_merchants[uid].add(mid)
            m_count[mid]   += 1
            m_val_sum[mid]  += v
            m_val_sq[mid]   += v * v
            m_users[mid].add(uid)
            edge_cnt[(uid, mid)] += 1

            ts = parse_ts_yelp(rv.get('date', ''))
            if not np.isnan(ts):
                if ts < u_min_ts[uid]: u_min_ts[uid] = ts
                if ts > u_max_ts[uid]: u_max_ts[uid] = ts
                if ts > global_max_ts: global_max_ts = ts
            n_kept += 1

    print(f'  Processed {n_kept:,} reviews')

    # --- Derived ---
    SECS_PER_DAY = 86400.0
    SECS_PER_YEAR = 365.0 * SECS_PER_DAY

    u_avg_val       = np.where(u_count > 0, u_val_sum / u_count, 0.0).astype(np.float32)
    u_var           = np.maximum(u_val_sq / np.maximum(u_count, 1) - (u_val_sum / np.maximum(u_count, 1))**2, 0.0)
    u_val_std       = np.sqrt(u_var).astype(np.float32)
    u_unique_m      = np.array([len(s) for s in u_merchants], dtype=np.float32)
    u_span_days     = np.where(np.isfinite(u_min_ts) & np.isfinite(u_max_ts),
                               (u_max_ts - u_min_ts) / SECS_PER_DAY, 0.0).astype(np.float32)
    u_recency       = np.where(np.isfinite(u_max_ts),
                               np.exp(-np.maximum(0.0, global_max_ts - u_max_ts) / SECS_PER_YEAR),
                               0.0).astype(np.float32)
    u_txns_per_week = (u_count.astype(np.float32) * 7.0 / np.maximum(u_span_days, 1.0))
    u_repeat_rate   = (u_count.astype(np.float32) / np.maximum(u_unique_m, 1.0))

    m_avg_val       = np.where(m_count > 0, m_val_sum / m_count, 0.0).astype(np.float32)
    m_var           = np.maximum(m_val_sq / np.maximum(m_count, 1) - (m_val_sum / np.maximum(m_count, 1))**2, 0.0)
    m_val_std       = np.sqrt(m_var).astype(np.float32)
    m_unique_u      = np.array([len(s) for s in m_users], dtype=np.float32)
    m_txns_per_u    = (m_count.astype(np.float32) / np.maximum(m_unique_u, 1.0))

    m_repeat_u      = np.zeros(merchantnum, dtype=np.float32)
    for (uid, mid), cnt in edge_cnt.items():
        if cnt > 1:
            m_repeat_u[mid] += 1.0
    m_user_repeat_rate = m_repeat_u / np.maximum(m_unique_u, 1.0)

    m_count_order = np.argsort(m_count)
    m_pop_rank    = np.zeros(merchantnum, dtype=np.float32)
    m_pop_rank[m_count_order] = np.linspace(0.0, 1.0, merchantnum)

    user_feat_raw = np.stack([
        u_count.astype(np.float32), u_avg_val,   u_unique_m,    u_span_days,
        u_recency,                   u_txns_per_week, u_val_std, u_repeat_rate,
    ], axis=1)
    merch_feat_raw = np.stack([
        m_count.astype(np.float32), m_avg_val,  m_unique_u,    merchant_cat,
        m_txns_per_u,                m_val_std, m_pop_rank,    m_user_repeat_rate,
    ], axis=1)

    # --- Normalise (same logic as CSV path) ---
    SKIP_NORM_USER  = {1, 4}       # avg_value already [0.2,1], recency already [0,1]
    SKIP_NORM_MERCH = {1, 6, 7}    # avg_value, pop_rank, repeat_rate already [0,1]

    def log_minmax(arr):
        v = np.log1p(np.clip(arr, 0, None))
        mn, mx = v.min(), v.max()
        return ((v - mn) / max(mx - mn, 1e-8)).astype(np.float32)

    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        return ((arr - mn) / max(mx - mn, 1e-8)).astype(np.float32)

    user_norm  = user_feat_raw.copy()
    merch_norm = merch_feat_raw.copy()

    for col in range(8):
        if col not in SKIP_NORM_USER:
            user_norm[:, col] = minmax(user_feat_raw[:, col]) if col == 6 else log_minmax(user_feat_raw[:, col])
    for col in range(8):
        if col not in SKIP_NORM_MERCH:
            merch_norm[:, col] = minmax(merch_feat_raw[:, col]) if col == 5 else log_minmax(merch_feat_raw[:, col])

    edge_normalization = 'stars / 5.0'

    print(f'User   features: {user_norm.shape}  '
          f'min/max per col: {user_norm.min(0).round(3)} / {user_norm.max(0).round(3)}')
    print(f'Merch  features: {merch_norm.shape}  '
          f'min/max per col: {merch_norm.min(0).round(3)} / {merch_norm.max(0).round(3)}')

# ===========================================================================
# CSV PATH — finance-merchant and synthetic-merchant
# ===========================================================================
else:
    RAW_CANDIDATES = {
        'finance-merchant':   os.path.join(REPO_ROOT, 'datasetRaw', 'finance', 'transactions_data.csv'),
        'synthetic-merchant': os.path.join(REPO_ROOT, 'datasetRaw', 'synthetic', 'dataset.csv'),
    }
    RAW_PATH = RAW_CANDIDATES[args.dataset]
    if not os.path.isfile(RAW_PATH):
        raise FileNotFoundError(f'Raw CSV not found: {RAW_PATH}')

    if args.dataset == 'finance-merchant':
        USER_COL, ITEM_COL, DATE_COL, AMOUNT_COL, MCC_COL = \
            'client_id', 'merchant_id', 'date', 'amount', 'mcc'
    else:
        USER_COL, ITEM_COL, DATE_COL, AMOUNT_COL, MCC_COL = \
            'customer_id', 'merchant_name', 'timestamp', 'amount_mnt', 'merchant_category_code'

    USE_COLS = [USER_COL, ITEM_COL, DATE_COL, AMOUNT_COL, MCC_COL]

    # --- Pass 1: p75 of log|amount| ---
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
            chunk[USER_COL].isin(user2id) & chunk[ITEM_COL].isin(merchant2id)
        ].dropna(subset=[AMOUNT_COL])
        all_amounts.append(chunk[AMOUNT_COL].abs().clip(lower=0).values)

    amounts_flat = np.concatenate(all_amounts)
    log_amounts  = np.log1p(amounts_flat)
    p75          = float(max(np.percentile(log_amounts, 75), 1.0))
    print(f'  p75(log|amount|) = {p75:.4f}')

    def norm_amount(amt: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.log1p(np.clip(amt, 0, None)) / p75))

    # --- Pass 2: accumulate statistics ---
    print('Pass 2: accumulating statistics ...')

    u_count     = np.zeros(usrnum,      dtype=np.int64)
    u_val_sum   = np.zeros(usrnum,      dtype=np.float64)
    u_val_sq    = np.zeros(usrnum,      dtype=np.float64)
    u_merchants = [set() for _ in range(usrnum)]
    u_min_ts    = np.full(usrnum,  np.inf)
    u_max_ts    = np.full(usrnum, -np.inf)

    m_count     = np.zeros(merchantnum, dtype=np.int64)
    m_val_sum   = np.zeros(merchantnum, dtype=np.float64)
    m_val_sq    = np.zeros(merchantnum, dtype=np.float64)
    m_users     = [set() for _ in range(merchantnum)]
    m_mcc: dict = {}

    edge_cnt: dict  = defaultdict(int)
    global_max_ts   = -np.inf

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
            chunk[USER_COL].isin(user2id) & chunk[ITEM_COL].isin(merchant2id)
        ].dropna(subset=[AMOUNT_COL, DATE_COL])

        chunk['uid']      = chunk[USER_COL].map(user2id)
        chunk['mid']      = chunk[ITEM_COL].map(merchant2id)
        chunk['norm_val'] = norm_amount(chunk[AMOUNT_COL].abs().values)
        chunk['unix_ts']  = chunk[DATE_COL].astype(np.int64) // 10**9

        iter_cols = ['uid', 'mid', 'norm_val', 'unix_ts', MCC_COL]
        for row in chunk[iter_cols].itertuples(index=False):
            uid, mid = int(row.uid), int(row.mid)
            v  = float(row.norm_val)
            ts = int(row.unix_ts)

            u_count[uid]   += 1
            u_val_sum[uid]  += v
            u_val_sq[uid]   += v * v
            u_merchants[uid].add(mid)
            if ts < u_min_ts[uid]: u_min_ts[uid] = ts
            if ts > u_max_ts[uid]: u_max_ts[uid] = ts
            if ts > global_max_ts: global_max_ts = ts

            m_count[mid]   += 1
            m_val_sum[mid]  += v
            m_val_sq[mid]   += v * v
            m_users[mid].add(uid)

            mcc_col_name = MCC_COL.replace('-', '_')
            mcc_val = getattr(row, mcc_col_name, None)
            if mid not in m_mcc and mcc_val is not None and pd.notna(mcc_val):
                m_mcc[mid] = int(mcc_val)

            edge_cnt[(uid, mid)] += 1

    print(f'  Total events: {u_count.sum():,}')

    # --- Derived ---
    SECS_PER_DAY  = 86400.0
    SECS_PER_YEAR = 365.0 * SECS_PER_DAY

    u_avg_val       = np.where(u_count > 0, u_val_sum / u_count, 0.0).astype(np.float32)
    u_var           = np.maximum(u_val_sq / np.maximum(u_count, 1) - (u_val_sum / np.maximum(u_count, 1))**2, 0.0)
    u_val_std       = np.sqrt(u_var).astype(np.float32)
    u_unique_m      = np.array([len(s) for s in u_merchants], dtype=np.float32)
    u_span_days     = np.where(np.isfinite(u_min_ts) & np.isfinite(u_max_ts),
                               (u_max_ts - u_min_ts) / SECS_PER_DAY, 0.0).astype(np.float32)
    u_recency       = np.where(np.isfinite(u_max_ts),
                               np.exp(-np.maximum(0.0, global_max_ts - u_max_ts) / SECS_PER_YEAR),
                               0.0).astype(np.float32)
    u_txns_per_week = (u_count.astype(np.float32) * 7.0 / np.maximum(u_span_days, 1.0))
    u_repeat_rate   = (u_count.astype(np.float32) / np.maximum(u_unique_m, 1.0))

    m_avg_val       = np.where(m_count > 0, m_val_sum / m_count, 0.0).astype(np.float32)
    m_var           = np.maximum(m_val_sq / np.maximum(m_count, 1) - (m_val_sum / np.maximum(m_count, 1))**2, 0.0)
    m_val_std       = np.sqrt(m_var).astype(np.float32)
    m_unique_u      = np.array([len(s) for s in m_users], dtype=np.float32)
    m_txns_per_u    = (m_count.astype(np.float32) / np.maximum(m_unique_u, 1.0))

    unique_mccs = sorted(set(m_mcc.values())) if m_mcc else [0]
    mcc2idx     = {v: i for i, v in enumerate(unique_mccs)}
    m_cat       = np.array(
        [float(mcc2idx.get(m_mcc.get(mid, 0), 0)) for mid in range(merchantnum)],
        dtype=np.float32)

    m_repeat_u  = np.zeros(merchantnum, dtype=np.float32)
    for (uid, mid), cnt in edge_cnt.items():
        if cnt > 1:
            m_repeat_u[mid] += 1.0
    m_user_repeat_rate = m_repeat_u / np.maximum(m_unique_u, 1.0)

    m_count_order = np.argsort(m_count)
    m_pop_rank    = np.zeros(merchantnum, dtype=np.float32)
    m_pop_rank[m_count_order] = np.linspace(0.0, 1.0, merchantnum)

    user_feat_raw = np.stack([
        u_count.astype(np.float32), u_avg_val,   u_unique_m,    u_span_days,
        u_recency,                   u_txns_per_week, u_val_std, u_repeat_rate,
    ], axis=1)
    merch_feat_raw = np.stack([
        m_count.astype(np.float32), m_avg_val,  m_unique_u,    m_cat,
        m_txns_per_u,                m_val_std, m_pop_rank,    m_user_repeat_rate,
    ], axis=1)

    # --- Normalise ---
    SKIP_NORM_USER  = {1, 4}
    SKIP_NORM_MERCH = {1, 6, 7}

    def log_minmax(arr):
        v = np.log1p(np.clip(arr, 0, None))
        mn, mx = v.min(), v.max()
        return ((v - mn) / max(mx - mn, 1e-8)).astype(np.float32)

    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        return ((arr - mn) / max(mx - mn, 1e-8)).astype(np.float32)

    user_norm  = user_feat_raw.copy()
    merch_norm = merch_feat_raw.copy()

    for col in range(8):
        if col not in SKIP_NORM_USER:
            user_norm[:, col] = minmax(user_feat_raw[:, col]) if col == 6 else log_minmax(user_feat_raw[:, col])
    for col in range(8):
        if col not in SKIP_NORM_MERCH:
            merch_norm[:, col] = minmax(merch_feat_raw[:, col]) if col == 5 else log_minmax(merch_feat_raw[:, col])

    edge_normalization = 'sigmoid(log1p(|amount|) / p75)'

    print(f'User   features: {user_norm.shape}  '
          f'min/max per col: {user_norm.min(0).round(3)} / {user_norm.max(0).round(3)}')
    print(f'Merch  features: {merch_norm.shape}  '
          f'min/max per col: {merch_norm.min(0).round(3)} / {merch_norm.max(0).round(3)}')


# ===========================================================================
# Save — same for all datasets
# ===========================================================================
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
    'edge_normalization':  edge_normalization,
    'dataset':             args.dataset,
    'schema':              '8+8',
}
with open(OUT_DIR + 'feature_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nSaved:')
for fname in ['user_features.npy', 'merchant_features.npy', 'feature_meta.json']:
    path = OUT_DIR + fname
    sz   = os.path.getsize(path) / 1024
    print(f'  {fname:<35} {sz:.0f} KB')
print('Done.')
