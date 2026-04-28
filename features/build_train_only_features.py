import argparse
import ast
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


DATASET_CFG = {
    'yelp-merchant': {
        'train_csv': 'train_yelp_merchant.csv',
        'raw_kind': 'yelp',
        'raw_path': os.path.join('datasetRaw', 'yelp', 'yelp_academic_dataset_review.json'),
        'category_path': os.path.join('datasetRaw', 'yelp', 'yelp_academic_dataset_business.json'),
    },
    'synthetic-merchant': {
        'train_csv': 'train_synthetic_merchant.csv',
        'raw_kind': 'synthetic',
        'raw_path': os.path.join('datasetRaw', 'synthetic', 'dataset.csv'),
    },
    'finance-merchant': {
        'train_csv': 'train_finance_merchant.csv',
        'raw_kind': 'finance',
        'raw_path': os.path.join('datasetRaw', 'finance', 'transactions_data.csv'),
    },
}

DEFAULT_GROUPS = ['value', 'time', 'category', 'repeat']
GROUP_ORDER = ['value', 'time', 'category', 'repeat', 'degree']
SECS_PER_DAY = 86400.0
SECS_PER_YEAR = 365.0 * SECS_PER_DAY


def minmax(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def log_minmax(arr):
    return minmax(np.log1p(np.clip(np.asarray(arr, dtype=np.float64), 0.0, None)))


def popularity_rank(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    order = np.argsort(-values, kind='mergesort')
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return ranks


def normalize_amount(values, p75):
    values = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.log1p(np.clip(values, 0.0, None)) / p75))).astype(np.float32)


def parse_yelp_ts(value):
    try:
        return int(datetime.strptime(value, '%Y-%m-%d %H:%M:%S').timestamp())
    except Exception:
        return None


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_train_counter(dataset_dir, train_csv):
    path = os.path.join(dataset_dir, train_csv)
    df = pd.read_csv(path, sep='\t')
    required = {'user_id', 'merchant_id', 'time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')

    # Preprocessing exports users as 1-indexed and merchants as 0-indexed.
    uids = df['user_id'].astype(np.int64).to_numpy() - 1
    mids = df['merchant_id'].astype(np.int64).to_numpy()
    tss = df['time'].astype(np.int64).to_numpy()
    counter = Counter(zip(uids.tolist(), mids.tolist(), tss.tolist()))
    return counter, len(df)


def load_yelp_category_matrix(root, dataset_dir, merchant2id, category_path):
    business_path = os.path.join(root, category_path)
    merchant_categories = {}
    vocab = set(['Unknown'])
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            bid = obj.get('business_id')
            if bid not in merchant2id:
                continue
            cats = obj.get('categories') or ''
            tokens = [c.strip() for c in cats.split(',') if c.strip()]
            if not tokens:
                tokens = ['Unknown']
            merchant_categories[merchant2id[bid]] = tokens
            vocab.update(tokens)

    names = sorted(vocab)
    cat2idx = {c: i for i, c in enumerate(names)}
    mat = np.zeros((len(merchant2id), len(names)), dtype=np.float32)
    for mid, cats in merchant_categories.items():
        for cat in cats:
            mat[mid, cat2idx.get(cat, cat2idx['Unknown'])] = 1.0
    unknown_idx = cat2idx['Unknown']
    empty = np.where(mat.sum(axis=1) == 0)[0]
    if len(empty):
        mat[empty, unknown_idx] = 1.0
    return mat, [f'category::{name}' for name in names]


def build_category_matrix_from_codes(num_items, merchant_codes):
    vocab = sorted({str(v) for v in merchant_codes.values() if v is not None})
    if 'Unknown' not in vocab:
        vocab.append('Unknown')
    code2idx = {c: i for i, c in enumerate(vocab)}
    unknown_idx = code2idx['Unknown']
    mat = np.zeros((num_items, len(vocab)), dtype=np.float32)
    for mid in range(num_items):
        code = merchant_codes.get(mid)
        idx = code2idx.get(str(code), unknown_idx)
        mat[mid, idx] = 1.0
    return mat, [f'category::{name}' for name in vocab]


def collect_yelp_events(root, cfg, user2id, merchant2id, train_counter):
    raw_path = os.path.join(root, cfg['raw_path'])
    remaining = train_counter.copy()
    events = []

    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not remaining:
                break
            rv = json.loads(line)
            uid_raw = rv.get('user_id')
            mid_raw = rv.get('business_id')
            if uid_raw not in user2id or mid_raw not in merchant2id:
                continue
            ts = parse_yelp_ts(rv.get('date', ''))
            if ts is None:
                continue
            uid = user2id[uid_raw]
            mid = merchant2id[mid_raw]
            key = (uid, mid, ts)
            if remaining.get(key, 0) <= 0:
                continue
            stars = rv.get('stars')
            if stars is None:
                continue
            value = float(stars) / 5.0
            events.append((uid, mid, ts, value, value * 5.0, None))
            remaining[key] -= 1
            if remaining[key] <= 0:
                del remaining[key]

    return events, remaining


def collect_tabular_events(root, cfg, user2id, merchant2id, train_counter):
    raw_path = os.path.join(root, cfg['raw_path'])
    kind = cfg['raw_kind']
    remaining = train_counter.copy()
    events_raw = []
    merchant_codes = {}

    if kind == 'synthetic':
        sample = pd.read_csv(raw_path, nrows=5)
        has_mcc = 'merchant_category_code' in sample.columns
        use_cols = ['customer_id', 'merchant_name', 'timestamp', 'amount_mnt']
        if has_mcc:
            use_cols.append('merchant_category_code')
        for chunk in pd.read_csv(raw_path, chunksize=500_000, low_memory=False, usecols=use_cols):
            if not remaining:
                break
            chunk['customer_id'] = chunk['customer_id'].astype(str)
            chunk['merchant_name'] = chunk['merchant_name'].astype(str)
            chunk = chunk[
                chunk['customer_id'].isin(user2id)
                & chunk['merchant_name'].isin(merchant2id)
            ].copy()
            if chunk.empty:
                continue
            chunk['ts'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
            chunk = chunk.dropna(subset=['ts', 'amount_mnt'])
            chunk['unix_ts'] = chunk['ts'].astype(np.int64) // 10**9
            chunk['uid'] = chunk['customer_id'].map(user2id)
            chunk['mid'] = chunk['merchant_name'].map(merchant2id)
            chunk['amount_raw'] = pd.to_numeric(chunk['amount_mnt'], errors='coerce').clip(lower=0)
            iter_cols = ['uid', 'mid', 'unix_ts', 'amount_raw']
            if has_mcc:
                iter_cols.append('merchant_category_code')
            for row in chunk[iter_cols].itertuples(index=False):
                uid, mid, ts = int(row.uid), int(row.mid), int(row.unix_ts)
                key = (uid, mid, ts)
                if remaining.get(key, 0) <= 0:
                    continue
                code = getattr(row, 'merchant_category_code', None) if has_mcc else None
                if code is not None and pd.notna(code):
                    merchant_codes[mid] = int(code)
                amt = float(row.amount_raw)
                events_raw.append((uid, mid, ts, amt, code))
                remaining[key] -= 1
                if remaining[key] <= 0:
                    del remaining[key]

    elif kind == 'finance':
        use_cols = ['client_id', 'merchant_id', 'date', 'amount', 'mcc']
        for chunk in pd.read_csv(raw_path, chunksize=500_000, low_memory=False, usecols=use_cols):
            if not remaining:
                break
            chunk['client_id'] = chunk['client_id'].astype(str)
            chunk['merchant_id'] = chunk['merchant_id'].astype(str)
            chunk = chunk[
                chunk['client_id'].isin(user2id)
                & chunk['merchant_id'].isin(merchant2id)
            ].copy()
            if chunk.empty:
                continue
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk['amount_raw'] = pd.to_numeric(
                chunk['amount'].astype(str).str.replace('$', '', regex=False),
                errors='coerce',
            ).abs()
            chunk = chunk.dropna(subset=['date', 'amount_raw'])
            chunk['unix_ts'] = chunk['date'].astype(np.int64) // 10**9
            chunk['uid'] = chunk['client_id'].map(user2id)
            chunk['mid'] = chunk['merchant_id'].map(merchant2id)
            for row in chunk[['uid', 'mid', 'unix_ts', 'amount_raw', 'mcc']].itertuples(index=False):
                uid, mid, ts = int(row.uid), int(row.mid), int(row.unix_ts)
                key = (uid, mid, ts)
                if remaining.get(key, 0) <= 0:
                    continue
                if pd.notna(row.mcc):
                    merchant_codes[mid] = int(row.mcc)
                amt = float(row.amount_raw)
                events_raw.append((uid, mid, ts, amt, row.mcc))
                remaining[key] -= 1
                if remaining[key] <= 0:
                    del remaining[key]
    else:
        raise ValueError(f'Unsupported tabular raw kind: {kind}')

    if events_raw:
        amounts = np.asarray([e[3] for e in events_raw], dtype=np.float64)
        p75 = float(max(np.percentile(np.log1p(amounts), 75), 1.0))
    else:
        p75 = 1.0

    norm_values = normalize_amount([e[3] for e in events_raw], p75) if events_raw else []
    events = []
    for idx, (uid, mid, ts, amount, code) in enumerate(events_raw):
        events.append((uid, mid, ts, float(norm_values[idx]), amount, code))
    return events, remaining, merchant_codes, p75


def add_neutral_missing(events, remaining, neutral_value=0.5):
    missing_events = []
    for (uid, mid, ts), count in remaining.items():
        for _ in range(count):
            missing_events.append((uid, mid, ts, neutral_value, 0.0, None))
    return events + missing_events


def build_features(events, num_users, num_items, merchant_cat, category_names, dataset, amount_p75=None):
    u_count = np.zeros(num_users, dtype=np.float64)
    u_val_sum = np.zeros(num_users, dtype=np.float64)
    u_val_sq = np.zeros(num_users, dtype=np.float64)
    u_min_ts = np.full(num_users, np.inf)
    u_max_ts = np.full(num_users, -np.inf)

    m_count = np.zeros(num_items, dtype=np.float64)
    m_val_sum = np.zeros(num_items, dtype=np.float64)
    m_val_sq = np.zeros(num_items, dtype=np.float64)
    m_min_ts = np.full(num_items, np.inf)
    m_max_ts = np.full(num_items, -np.inf)

    edge_count = defaultdict(int)
    edge_val_sum = defaultdict(float)
    edge_raw_sum = defaultdict(float)
    global_max_ts = max((e[2] for e in events), default=0)

    for uid, mid, ts, value, raw_value, _ in events:
        u_count[uid] += 1.0
        u_val_sum[uid] += value
        u_val_sq[uid] += value * value
        u_min_ts[uid] = min(u_min_ts[uid], ts)
        u_max_ts[uid] = max(u_max_ts[uid], ts)

        m_count[mid] += 1.0
        m_val_sum[mid] += value
        m_val_sq[mid] += value * value
        m_min_ts[mid] = min(m_min_ts[mid], ts)
        m_max_ts[mid] = max(m_max_ts[mid], ts)

        key = (uid, mid)
        edge_count[key] += 1
        edge_val_sum[key] += value
        edge_raw_sum[key] += raw_value

    user_unique = np.zeros(num_users, dtype=np.float64)
    merchant_unique = np.zeros(num_items, dtype=np.float64)
    user_repeat = np.zeros(num_users, dtype=np.float64)
    merchant_repeat = np.zeros(num_items, dtype=np.float64)
    user_cat_sum = np.zeros((num_users, merchant_cat.shape[1]), dtype=np.float32)

    for (uid, mid), cnt in edge_count.items():
        user_unique[uid] += 1.0
        merchant_unique[mid] += 1.0
        if cnt > 1:
            user_repeat[uid] += 1.0
            merchant_repeat[mid] += 1.0
        user_cat_sum[uid] += merchant_cat[mid] * float(cnt)

    u_avg = np.divide(u_val_sum, np.maximum(u_count, 1.0)).astype(np.float32)
    m_avg = np.divide(m_val_sum, np.maximum(m_count, 1.0)).astype(np.float32)
    u_std = np.sqrt(np.maximum(u_val_sq / np.maximum(u_count, 1.0) - u_avg.astype(np.float64) ** 2, 0.0))
    m_std = np.sqrt(np.maximum(m_val_sq / np.maximum(m_count, 1.0) - m_avg.astype(np.float64) ** 2, 0.0))

    u_span_days = np.where(np.isfinite(u_min_ts) & np.isfinite(u_max_ts), (u_max_ts - u_min_ts) / SECS_PER_DAY, 0.0)
    m_span_days = np.where(np.isfinite(m_min_ts) & np.isfinite(m_max_ts), (m_max_ts - m_min_ts) / SECS_PER_DAY, 0.0)
    u_recency = np.where(np.isfinite(u_max_ts), np.exp(-np.maximum(0.0, global_max_ts - u_max_ts) / SECS_PER_YEAR), 0.0)
    m_recency = np.where(np.isfinite(m_max_ts), np.exp(-np.maximum(0.0, global_max_ts - m_max_ts) / SECS_PER_YEAR), 0.0)

    user_repeat_rate = np.divide(user_repeat, np.maximum(user_unique, 1.0)).astype(np.float32)
    merchant_repeat_rate = np.divide(merchant_repeat, np.maximum(merchant_unique, 1.0)).astype(np.float32)
    user_cat = np.divide(user_cat_sum, np.maximum(u_count[:, None], 1.0)).astype(np.float32)

    user_blocks = {
        'value': (
            np.stack([u_avg, minmax(u_std)], axis=1),
            ['avg_interaction_value', 'value_std_norm'],
        ),
        'time': (
            np.stack([log_minmax(u_span_days), u_recency.astype(np.float32)], axis=1),
            ['activity_span_days', 'recency_score'],
        ),
        'category': (user_cat, category_names),
        'repeat': (
            user_repeat_rate.reshape(-1, 1),
            ['repeat_merchant_rate'],
        ),
        'degree': (
            np.stack([
                log_minmax(u_count),
                log_minmax(user_unique),
                popularity_rank(u_count),
            ], axis=1),
            ['event_count_norm', 'unique_merchant_count_norm', 'activity_rank_norm'],
        ),
    }

    merchant_blocks = {
        'value': (
            np.stack([m_avg, minmax(m_std)], axis=1),
            ['avg_interaction_value', 'value_std_norm'],
        ),
        'time': (
            np.stack([log_minmax(m_span_days), m_recency.astype(np.float32)], axis=1),
            ['activity_span_days', 'recency_score'],
        ),
        'category': (merchant_cat.astype(np.float32), category_names),
        'repeat': (
            merchant_repeat_rate.reshape(-1, 1),
            ['user_repeat_rate'],
        ),
        'degree': (
            np.stack([
                log_minmax(m_count),
                log_minmax(merchant_unique),
                popularity_rank(m_count),
            ], axis=1),
            ['event_count_norm', 'unique_user_count_norm', 'popularity_rank_norm'],
        ),
    }

    user_features, user_names, user_groups = concat_blocks(user_blocks)
    merchant_features, merchant_names, merchant_groups = concat_blocks(merchant_blocks)
    edge_weights, edge_meta = build_edge_weights(dataset, edge_count, edge_val_sum, edge_raw_sum)
    if amount_p75 is not None:
        edge_meta['amount_log_p75_for_node_value'] = amount_p75

    meta = {
        'version': 2,
        'source': 'train_split_only',
        'dataset': dataset,
        'schema': 'train-only grouped node features',
        'default_node_feature_groups': DEFAULT_GROUPS,
        'group_order': GROUP_ORDER,
        'degree_excluded_from_default': True,
        'user_feature_names': user_names,
        'merchant_feature_names': merchant_names,
        'user_feature_groups': user_groups,
        'merchant_feature_groups': merchant_groups,
        'user_value_feature_names': ['avg_interaction_value', 'value_std_norm'],
        'merchant_value_feature_names': ['avg_interaction_value', 'value_std_norm'],
        'edge_feature_names': [edge_meta['edge_reweight_strategy']],
        'category_feature_count': len(category_names),
        'train_event_count': int(len(events)),
        'train_pair_count': int(len(edge_count)),
        'feature_selection': (
            'default all=value+time+category+repeat; degree/popularity available '
            'only via --node_feature_groups degree or all_plus_degree'
        ),
    }
    meta.update(edge_meta)
    return user_features, merchant_features, edge_weights, meta


def concat_blocks(blocks):
    arrays = []
    names = []
    groups = {}
    cursor = 0
    for group in GROUP_ORDER:
        arr, group_names = blocks[group]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = arr.astype(np.float32)
        arrays.append(arr)
        full_names = [f'{group}:{name}' for name in group_names]
        names.extend(full_names)
        groups[group] = list(range(cursor, cursor + arr.shape[1]))
        cursor += arr.shape[1]
    return np.concatenate(arrays, axis=1), names, groups


def build_edge_weights(dataset, edge_count, edge_val_sum, edge_raw_sum):
    pairs = list(edge_count.keys())
    if not pairs:
        return {}, {'edge_reweight_strategy': 'none', 'edge_weight_range': [0.0, 0.0]}

    if dataset == 'yelp-merchant':
        means = np.asarray([edge_val_sum[p] / edge_count[p] * 5.0 for p in pairs], dtype=np.float64)
        global_mu = float(np.mean(means))
        c = 10.0
        counts = np.asarray([edge_count[p] for p in pairs], dtype=np.float64)
        shrunk = (counts * means + c * global_mu) / (counts + c)
        mn, mx = float(np.min(shrunk)), float(np.max(shrunk))
        weights = (shrunk - mn) / max(mx - mn, 1e-8)
        meta = {
            'edge_reweight_strategy': 'bayesian_shrinkage_stars_train_only',
            'edge_reweight_pseudo_count': c,
            'edge_reweight_global_mean': global_mu,
            'edge_raw_value_range': [mn, mx],
        }
    else:
        raw_means = np.asarray([edge_raw_sum[p] / max(edge_count[p], 1) for p in pairs], dtype=np.float64)
        log_amt = np.log1p(np.clip(raw_means, 0.0, None))
        order = np.argsort(np.argsort(log_amt, kind='mergesort'), kind='mergesort')
        n = len(order)
        weights = (order.astype(np.float64) + 1.0) / (n + 1.0)
        meta = {
            'edge_reweight_strategy': 'log_quantile_rank_amount_train_only',
        }
    weights = np.clip(weights, 1e-3, 1.0)
    meta['edge_weight_range'] = [float(np.min(weights)), float(np.max(weights))]
    return {pairs[i]: float(weights[i]) for i in range(len(pairs))}, meta


def build_dataset(root, dataset, min_match_rate):
    cfg = DATASET_CFG[dataset]
    dataset_dir = os.path.join(root, 'Datasets', dataset)
    user2id = read_pickle(os.path.join(dataset_dir, 'user2id.pkl'))
    merchant2id = read_pickle(os.path.join(dataset_dir, 'merchant2id.pkl'))
    train_counter, train_events = load_train_counter(dataset_dir, cfg['train_csv'])

    print(f'[{dataset}] users={len(user2id):,} merchants={len(merchant2id):,} train_events={train_events:,}')
    if cfg['raw_kind'] == 'yelp':
        merchant_cat, category_names = load_yelp_category_matrix(root, dataset_dir, merchant2id, cfg['category_path'])
        events, remaining = collect_yelp_events(root, cfg, user2id, merchant2id, train_counter)
        amount_p75 = None
    else:
        events, remaining, merchant_codes, amount_p75 = collect_tabular_events(root, cfg, user2id, merchant2id, train_counter)
        merchant_cat, category_names = build_category_matrix_from_codes(len(merchant2id), merchant_codes)

    matched = train_events - sum(remaining.values())
    match_rate = matched / max(train_events, 1)
    print(f'[{dataset}] matched raw values: {matched:,}/{train_events:,} ({match_rate:.2%})')
    if remaining:
        print(f'[{dataset}] warning: {sum(remaining.values()):,} train rows were not matched; filling neutral values')
        events = add_neutral_missing(events, remaining)
    if match_rate < min_match_rate:
        raise RuntimeError(
            f'{dataset} matched only {match_rate:.2%}; refusing to write features. '
            f'Check timestamp/mapping alignment or lower --min-match-rate.'
        )

    user_features, merchant_features, edge_weights, meta = build_features(
        events=events,
        num_users=len(user2id),
        num_items=len(merchant2id),
        merchant_cat=merchant_cat,
        category_names=category_names,
        dataset=dataset,
        amount_p75=amount_p75,
    )
    meta['train_csv'] = cfg['train_csv']
    meta['raw_value_source'] = cfg['raw_path']
    meta['raw_match_rate'] = match_rate

    np.save(os.path.join(dataset_dir, 'user_features.npy'), user_features)
    np.save(os.path.join(dataset_dir, 'merchant_features.npy'), merchant_features)
    with open(os.path.join(dataset_dir, 'edge_weights.pkl'), 'wb') as f:
        pickle.dump(edge_weights, f)
    with open(os.path.join(dataset_dir, 'feature_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'[{dataset}] wrote user_features.npy {user_features.shape}')
    print(f'[{dataset}] wrote merchant_features.npy {merchant_features.shape}')
    print(f'[{dataset}] wrote edge_weights.pkl {len(edge_weights):,} pairs')
    print(f'[{dataset}] groups: default={DEFAULT_GROUPS}, extra=degree/all_plus_degree')


def main():
    parser = argparse.ArgumentParser(description='Build train-only SelfGNN node and edge features.')
    parser.add_argument('--data', default='all',
                        choices=['all'] + sorted(DATASET_CFG.keys()),
                        help='dataset to build')
    parser.add_argument('--root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                        help='repository root')
    parser.add_argument('--min-match-rate', type=float, default=0.98,
                        help='minimum raw-value match rate required before writing features')
    args = parser.parse_args()

    datasets = sorted(DATASET_CFG.keys()) if args.data == 'all' else [args.data]
    for dataset in datasets:
        build_dataset(os.path.abspath(args.root), dataset, args.min_match_rate)


if __name__ == '__main__':
    main()
