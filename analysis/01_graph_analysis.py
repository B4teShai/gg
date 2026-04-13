"""
Script 1: Graph construction + structural analysis for merchant recommendation.
Writes analysis/artifacts/graph_stats.json + 6 PNGs in analysis/plots/.
Runtime: ~5-10 min.

Uniform 4D user + 4D merchant feature schema across all datasets:
  User:     [interaction_count, avg_interaction_value, unique_merchant_count, activity_span_days]
  Merchant: [interaction_count, avg_interaction_value, unique_user_count, category_id]
  Edge:     [normalized_interaction_value]  in (0,1)
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    ROOT, YELP_DIR, FIN_DIR, SYN_DIR,
    SAMPLE_ROWS, KCORE_K,
    save_artifact, sample_csv,
    kcore_filter_bipartite, build_csr_bipartite, graph_metrics,
    degree_distribution_plot, normalize_amount_to_edge_weight,
    USER_FEATURE_NAMES, MERCHANT_FEATURE_NAMES, EDGE_FEATURE_NAMES,
)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def _feat_stats(arr: np.ndarray) -> dict:
    """Return summary stats for a feature column (for reporting)."""
    return {
        'mean': float(np.nanmean(arr)),
        'std':  float(np.nanstd(arr)),
        'min':  float(np.nanmin(arr)),
        'max':  float(np.nanmax(arr)),
    }


def _uniform_feature_report(df_edges: pd.DataFrame,
                             user_col: str,
                             merchant_col: str,
                             value_col: str,
                             date_col: str,
                             category_col: str,
                             norm_fn) -> dict:
    """
    Compute uniform 4D user + 4D merchant feature statistics from an edge list.
    All computations are from the sampled edge list only (no external lookups).

    User features (4D):
      [0] interaction_count      — number of interactions per user
      [1] avg_interaction_value  — mean normalized edge value per user
      [2] unique_merchant_count  — number of distinct merchants per user
      [3] activity_span_days     — (last - first interaction date) in days

    Merchant features (4D):
      [0] interaction_count      — number of interactions per merchant
      [1] avg_interaction_value  — mean normalized edge value per merchant
      [2] unique_user_count      — number of distinct users per merchant
      [3] category_id            — integer-encoded category (mode per merchant)
    """
    df = df_edges.copy()
    df['norm_value'] = norm_fn(df[value_col])

    # ---- User features ----
    u_grp  = df.groupby(user_col)
    u_int  = u_grp.size().values                               # [0] interaction_count
    u_avg  = u_grp['norm_value'].mean().values                 # [1] avg_interaction_value
    u_umc  = u_grp[merchant_col].nunique().values              # [2] unique_merchant_count

    # [3] activity_span_days
    if date_col in df.columns and df[date_col].notna().any():
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        span = u_grp['_date'].agg(lambda x: (x.max() - x.min()).days)
        u_span = span.values.astype(float)
    else:
        u_span = np.zeros(len(u_int))

    # ---- Merchant features ----
    m_grp = df.groupby(merchant_col)
    m_int = m_grp.size().values                               # [0] interaction_count
    m_avg = m_grp['norm_value'].mean().values                 # [1] avg_interaction_value
    m_uuc = m_grp[user_col].nunique().values                  # [2] unique_user_count

    # [3] category_id (int-encoded merchant category)
    if category_col in df.columns and df[category_col].notna().any():
        df['_cat'] = df[category_col].astype('category').cat.codes
        m_cat = m_grp['_cat'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else -1).values
    else:
        m_cat = np.zeros(m_int.shape)

    return {
        'n_users':    int(len(u_int)),
        'n_merchants': int(len(m_int)),
        'user_features': {
            'names':    USER_FEATURE_NAMES,
            'dim':      4,
            'stats': {
                'interaction_count':     _feat_stats(u_int),
                'avg_interaction_value': _feat_stats(u_avg),
                'unique_merchant_count': _feat_stats(u_umc),
                'activity_span_days':    _feat_stats(u_span),
            },
        },
        'merchant_features': {
            'names':    MERCHANT_FEATURE_NAMES,
            'dim':      4,
            'stats': {
                'interaction_count':     _feat_stats(m_int),
                'avg_interaction_value': _feat_stats(m_avg),
                'unique_user_count':     _feat_stats(m_uuc),
                'category_id':           _feat_stats(m_cat),
            },
        },
        'edge_features': {
            'names': EDGE_FEATURE_NAMES,
            'dim':   1,
            'stats': _feat_stats(df['norm_value'].values),
        },
    }


# ------------------------------------------------------------------ #
#  Yelp                                                                #
# ------------------------------------------------------------------ #
def yelp_graph_analysis() -> dict:
    print('\n=== Yelp Graph Analysis (user-business recommendation) ===')

    review_path = os.path.join(YELP_DIR, 'yelp_academic_dataset_review.json')

    # Stream 500K reviews
    print(f'  Streaming {SAMPLE_ROWS:,} reviews...')
    users_list, merchants_list, stars_list, dates_list, cats_list = [], [], [], [], []
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
            users_list.append(r['user_id'])
            merchants_list.append(r['business_id'])
            stars_list.append(float(r.get('stars', 3.0)))
            dates_list.append(r.get('date', ''))
            count += 1

    df = pd.DataFrame({
        'user_id':     users_list,
        'merchant_id': merchants_list,
        'stars':       stars_list,
        'date':        dates_list,
        'category':    None,  # category requires business.json lookup — set to None
    })

    print(f'  Sampled: {len(df):,} edges | {df["user_id"].nunique():,} users | '
          f'{df["merchant_id"].nunique():,} merchants')

    # Edge weight: stars / 5.0  →  [0.2, 1.0]
    edge_weights = pd.Series(df['stars'].values / 5.0, index=df.index)

    # Before k-core
    B_raw, _, _ = build_csr_bipartite(df['user_id'], df['merchant_id'])
    before = graph_metrics(B_raw)
    before['note'] = f'500K sample, before k-core'

    # K-core
    print(f'  Applying k-core (k={KCORE_K})...')
    u_filt, m_filt = kcore_filter_bipartite(df['user_id'], df['merchant_id'], k=KCORE_K)
    mask_filt = u_filt.index
    df_filt   = df.iloc[mask_filt].reset_index(drop=True)
    w_filt    = edge_weights.iloc[mask_filt].reset_index(drop=True)

    print(f'  After k-core: {df_filt["user_id"].nunique():,} users | '
          f'{df_filt["merchant_id"].nunique():,} merchants | {len(df_filt):,} edges')

    B_kcore, _, _ = build_csr_bipartite(df_filt['user_id'], df_filt['merchant_id'],
                                         weights=w_filt)
    after = graph_metrics(B_kcore)
    after['note'] = f'500K sample, k-core k={KCORE_K}'

    degree_distribution_plot(B_kcore, 'yelp')

    # Uniform feature report
    norm_fn = lambda s: s / 5.0
    feat = _uniform_feature_report(
        df_filt, 'user_id', 'merchant_id', 'stars', 'date', 'category', norm_fn
    )

    # Ground truth from preprocess_bipartite_graph.ipynb (full dataset k-core)
    ground_truth = {
        'n_users':    268653,
        'n_merchants': 109325,
        'source':     'preprocess_bipartite_graph.ipynb (full dataset, k-core k=5)',
    }

    return {
        'dataset':            'yelp',
        'before_kcore':       before,
        'after_kcore':        after,
        'ground_truth_kcore': ground_truth,
        'uniform_features':   feat,
        'edge_weight_formula': 'stars / 5.0  →  [0.2, 1.0]',
        'temporal_slices':    5,
        'temporal_note':      '2005-2022, supports 5 equal time slices for SelfGNN',
        'recommendation_suitability': {
            'explicit_feedback':   True,
            'long_tail_structure': True,
            'social_graph':        True,
            'note': 'Best for heterogeneous recommendation; explicit star ratings give clean supervision signal.',
        },
    }


# ------------------------------------------------------------------ #
#  Finance                                                             #
# ------------------------------------------------------------------ #
def finance_graph_analysis() -> dict:
    print('\n=== Finance Graph Analysis (user-merchant spending recommendation) ===')

    txn_path = os.path.join(FIN_DIR, 'transactions_data.csv')

    print(f'  Reading {SAMPLE_ROWS:,} transactions...')
    df = sample_csv(txn_path, n=SAMPLE_ROWS,
                    usecols=['id', 'client_id', 'merchant_id', 'amount', 'date', 'mcc'])
    df = df.dropna(subset=['client_id', 'merchant_id'])
    df['client_id']   = df['client_id'].astype(str)
    df['merchant_id'] = df['merchant_id'].astype(str)
    df['amount']      = pd.to_numeric(
        df['amount'].astype(str).str.replace('$', '', regex=False),
        errors='coerce'
    )
    df = df.dropna(subset=['amount'])
    df['amount_abs'] = df['amount'].abs()

    print(f'  Sampled: {len(df):,} edges | {df["client_id"].nunique():,} users | '
          f'{df["merchant_id"].nunique():,} merchants')

    # Edge weight: normalize amount
    edge_weights = normalize_amount_to_edge_weight(df['amount_abs'])

    # Before k-core
    B_raw, _, _ = build_csr_bipartite(df['client_id'], df['merchant_id'])
    before = graph_metrics(B_raw)
    before['note'] = '500K sample, before k-core'

    # K-core
    print(f'  Applying k-core (k={KCORE_K})...')
    u_filt, m_filt = kcore_filter_bipartite(df['client_id'], df['merchant_id'], k=KCORE_K)
    mask_filt = u_filt.index
    df_filt   = df.iloc[mask_filt].reset_index(drop=True)
    w_filt    = edge_weights.iloc[mask_filt].reset_index(drop=True)

    print(f'  After k-core: {df_filt["client_id"].nunique():,} users | '
          f'{df_filt["merchant_id"].nunique():,} merchants | {len(df_filt):,} edges')

    B_kcore, _, _ = build_csr_bipartite(df_filt['client_id'], df_filt['merchant_id'],
                                         weights=w_filt)
    after = graph_metrics(B_kcore)
    after['note'] = f'500K sample, k-core k={KCORE_K}'

    degree_distribution_plot(B_kcore, 'finance')

    # Uniform feature report
    # Rename columns to match generic interface
    df_filt2 = df_filt.rename(columns={
        'client_id':   'user_id',
        'merchant_id': 'merchant_id',
        'amount_abs':  'value',
        'mcc':         'category',
    })
    norm_fn = normalize_amount_to_edge_weight
    feat = _uniform_feature_report(
        df_filt2, 'user_id', 'merchant_id', 'value', 'date', 'category', norm_fn
    )

    return {
        'dataset':          'finance',
        'before_kcore':     before,
        'after_kcore':      after,
        'uniform_features': feat,
        'edge_weight_formula': 'sigmoid(log1p(|amount|) / p75)  →  (0, 1)',
        'temporal_slices':  10,
        'temporal_note':    '2010-2019, ~10 annual time slices for SelfGNN',
        'recommendation_suitability': {
            'explicit_feedback':   False,
            'long_tail_structure': True,
            'social_graph':        False,
            'note': ('Implicit feedback (spending as preference proxy). '
                     'Long-tail merchant distribution makes this realistic for recommendation.'),
        },
    }


# ------------------------------------------------------------------ #
#  Synthetic                                                           #
# ------------------------------------------------------------------ #
def synthetic_graph_analysis() -> dict:
    print('\n=== Synthetic Graph Analysis (customer-merchant recommendation) ===')

    syn_path = os.path.join(SYN_DIR, 'dataset.csv')

    print(f'  Reading {SAMPLE_ROWS:,} rows...')
    df = sample_csv(syn_path, n=SAMPLE_ROWS,
                    usecols=['customer_id', 'merchant_name', 'amount_mnt',
                             'timestamp', 'merchant_category_code'])
    df = df.dropna(subset=['customer_id'])
    df['merchant_name'] = df['merchant_name'].fillna('UNKNOWN_MERCHANT')
    df['customer_id']   = df['customer_id'].astype(str)
    df['amount_mnt']    = pd.to_numeric(df['amount_mnt'], errors='coerce').fillna(0)

    n_merchants_raw = df['merchant_name'].nunique()
    print(f'  Sampled: {len(df):,} edges | {df["customer_id"].nunique():,} users | '
          f'{n_merchants_raw:,} merchants')
    if n_merchants_raw < 100:
        print(f'  NOTE: only {n_merchants_raw} unique merchants — graph will be near-dense')

    edge_weights = normalize_amount_to_edge_weight(df['amount_mnt'])

    # Before k-core
    B_raw, _, _ = build_csr_bipartite(df['customer_id'], df['merchant_name'])
    before = graph_metrics(B_raw)
    before['note'] = '500K sample, before k-core'

    # K-core
    print(f'  Applying k-core (k={KCORE_K})...')
    u_filt, m_filt = kcore_filter_bipartite(df['customer_id'], df['merchant_name'], k=KCORE_K)
    mask_filt = u_filt.index
    df_filt   = df.iloc[mask_filt].reset_index(drop=True)
    w_filt    = edge_weights.iloc[mask_filt].reset_index(drop=True)

    print(f'  After k-core: {df_filt["customer_id"].nunique():,} users | '
          f'{df_filt["merchant_name"].nunique():,} merchants | {len(df_filt):,} edges')

    B_kcore, _, _ = build_csr_bipartite(df_filt['customer_id'], df_filt['merchant_name'],
                                         weights=w_filt)
    after = graph_metrics(B_kcore)
    after['note']  = f'500K sample, k-core k={KCORE_K}'
    after['density_warning'] = (
        f'Near-dense graph ({after["density"]:.4f}) due to only {n_merchants_raw} merchants. '
        'Graph structure contributes less signal than Yelp/Finance for GNN evaluation.'
    )

    degree_distribution_plot(B_kcore, 'synthetic')

    # Uniform feature report
    df_filt2 = df_filt.rename(columns={
        'customer_id':            'user_id',
        'merchant_name':          'merchant_id',
        'amount_mnt':             'value',
        'timestamp':              'date',
        'merchant_category_code': 'category',
    })
    norm_fn = normalize_amount_to_edge_weight
    feat = _uniform_feature_report(
        df_filt2, 'user_id', 'merchant_id', 'value', 'date', 'category', norm_fn
    )

    return {
        'dataset':          'synthetic',
        'before_kcore':     before,
        'after_kcore':      after,
        'uniform_features': feat,
        'edge_weight_formula': 'sigmoid(log1p(amount) / p75)  →  (0, 1)',
        'temporal_slices':  23,
        'temporal_note':    '2023-2024, ~23 monthly time slices',
        'recommendation_suitability': {
            'explicit_feedback':   False,
            'long_tail_structure': False,
            'social_graph':        False,
            'note': (f'Only {n_merchants_raw} merchants creates near-complete bipartite graph. '
                     'Useful for controlled experiments and scalability tests. '
                     'Not representative of real-world recommendation sparsity.'),
        },
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print('Running graph analysis (merchant recommendation)...')
    graph_stats = {
        'yelp':      yelp_graph_analysis(),
        'finance':   finance_graph_analysis(),
        'synthetic': synthetic_graph_analysis(),
        'uniform_feature_schema': {
            'user_features':     USER_FEATURE_NAMES,
            'merchant_features': MERCHANT_FEATURE_NAMES,
            'edge_features':     EDGE_FEATURE_NAMES,
            'user_dim':          4,
            'merchant_dim':      4,
            'edge_dim':          1,
        },
    }
    save_artifact(graph_stats, 'graph_stats')
    print('\nDone. Artifact: analysis/artifacts/graph_stats.json')
    print('Plots: analysis/plots/')


if __name__ == '__main__':
    main()
