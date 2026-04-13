"""
Shared utilities for dataset comparison analysis.
No networkx — uses scipy.sparse for all graph operations.

Uniform feature schema (4D user + 4D merchant) for all datasets:
  User:     [interaction_count, avg_interaction_value, unique_merchants, activity_span_days]
  Merchant: [interaction_count, avg_interaction_value, unique_users, category_id]
  Edge:     [normalized_interaction_value]
"""
import os
import json
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
YELP_DIR     = os.path.join(ROOT, 'datasetRaw', 'yelp')
FIN_DIR      = os.path.join(ROOT, 'datasetRaw', 'finance')
SYN_DIR      = os.path.join(ROOT, 'datasetRaw', 'synthetic')
ARTIFACT_DIR = os.path.join(ROOT, 'analysis', 'artifacts')
PLOT_DIR     = os.path.join(ROOT, 'analysis', 'plots')

SAMPLE_ROWS  = 500_000
KCORE_K      = 5

# Uniform feature names (same across all datasets)
USER_FEATURE_NAMES     = ['interaction_count', 'avg_interaction_value',
                           'unique_merchant_count', 'activity_span_days']
MERCHANT_FEATURE_NAMES = ['interaction_count', 'avg_interaction_value',
                           'unique_user_count', 'category_id']
EDGE_FEATURE_NAMES     = ['normalized_interaction_value']

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  I/O helpers                                                         #
# ------------------------------------------------------------------ #
def save_artifact(data: dict, name: str) -> None:
    path = os.path.join(ARTIFACT_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  Saved artifact: {path}')


def load_artifact(name: str) -> dict:
    path = os.path.join(ARTIFACT_DIR, f'{name}.json')
    with open(path) as f:
        return json.load(f)


def count_lines(path: str) -> int:
    """Fast line count via wc -l."""
    result = subprocess.run(['wc', '-l', path], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])


def sample_csv(path: str, n: int = SAMPLE_ROWS, usecols: list = None) -> pd.DataFrame:
    """Read up to n rows from CSV. Strips leading '$' from currency columns."""
    df = pd.read_csv(path, nrows=n, usecols=usecols, low_memory=False)
    for col in df.columns:
        if df[col].dtype == object:
            sample_vals = df[col].dropna().head(10).tolist()
            if sample_vals and all(str(v).startswith('$') for v in sample_vals if str(v) != 'nan'):
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('$', '', regex=False),
                    errors='coerce'
                )
    return df


def normalize_amount_to_edge_weight(amounts: pd.Series) -> pd.Series:
    """
    Normalize transaction amounts to (0, 1) for edge weight.
    Uses sigmoid(log1p(|amount|) / p75) — comparable scale to Yelp stars/5.
    """
    log_amt = np.log1p(amounts.abs().clip(lower=0))
    p75 = float(log_amt.quantile(0.75))
    scale = max(p75, 1.0)  # avoid division by zero
    return pd.Series(1.0 / (1.0 + np.exp(-log_amt / scale)), index=amounts.index)


# ------------------------------------------------------------------ #
#  K-core filtering                                                    #
# ------------------------------------------------------------------ #
def kcore_filter_bipartite(
    user_col: pd.Series,
    merchant_col: pd.Series,
    k: int = KCORE_K
) -> tuple:
    """
    Iterative k-core filter on bipartite (user, merchant) edge list.
    Mirrors selfGNN-Feature/feature_extractor.py kcore_filter().
    Returns (filtered_user_col, filtered_merchant_col).
    """
    u = user_col.reset_index(drop=True)
    m = merchant_col.reset_index(drop=True)

    for _ in range(1000):
        u_deg = u.value_counts()
        m_deg = m.value_counts()

        valid_u = set(u_deg[u_deg >= k].index)
        valid_m = set(m_deg[m_deg >= k].index)

        mask  = u.isin(valid_u) & m.isin(valid_m)
        new_u = u[mask].reset_index(drop=True)
        new_m = m[mask].reset_index(drop=True)

        if len(new_u) == len(u):
            break
        u, m = new_u, new_m

    return u, m


# ------------------------------------------------------------------ #
#  Graph construction                                                  #
# ------------------------------------------------------------------ #
def build_csr_bipartite(
    users: pd.Series,
    merchants: pd.Series,
    weights: pd.Series = None
) -> tuple:
    """
    Build scipy.sparse CSR bipartite incidence matrix B (n_users x n_merchants).
    Returns (B, user2id, merchant2id).
    """
    u_cats = pd.Categorical(users)
    m_cats = pd.Categorical(merchants)
    user2id     = {v: k for k, v in enumerate(u_cats.categories)}
    merchant2id = {v: k for k, v in enumerate(m_cats.categories)}

    row  = u_cats.codes
    col  = m_cats.codes
    data = np.ones(len(row)) if weights is None else weights.values.astype(float)

    B = sp.csr_matrix(
        (data, (row, col)),
        shape=(len(u_cats.categories), len(m_cats.categories))
    )
    B.sum_duplicates()
    return B, user2id, merchant2id


# ------------------------------------------------------------------ #
#  Graph metrics                                                       #
# ------------------------------------------------------------------ #
def graph_metrics(B: sp.csr_matrix) -> dict:
    """
    Compute structural metrics for bipartite matrix B (n_users x n_merchants).
    Degrees are connection counts (binarized), not weight sums.
    Uses scipy only — no networkx.
    """
    n_u, n_m = B.shape
    n_edges  = B.nnz
    density  = n_edges / (n_u * n_m) if (n_u * n_m) > 0 else 0.0

    # Binarize for degree count (not weight sum)
    B_bin = B.copy()
    B_bin.data = np.ones_like(B_bin.data)
    user_deg     = np.array(B_bin.sum(axis=1)).flatten()
    merchant_deg = np.array(B_bin.sum(axis=0)).flatten()

    # Connected components via block adjacency [[0, B],[B^T, 0]]
    Z_uu = sp.csr_matrix((n_u, n_u))
    Z_mm = sp.csr_matrix((n_m, n_m))
    full = sp.bmat([[Z_uu, B], [B.T, Z_mm]], format='csr')
    full.data = np.ones_like(full.data)
    n_comp, labels = csgraph.connected_components(full, directed=False)
    cc_sizes   = np.bincount(labels)
    largest_cc = int(cc_sizes.max())

    return {
        'n_users':              int(n_u),
        'n_merchants':          int(n_m),
        'n_edges':              int(n_edges),
        'density':              float(density),
        'sparsity':             float(1.0 - density),
        'avg_user_degree':      float(user_deg.mean())     if len(user_deg)     > 0 else 0.0,
        'max_user_degree':      float(user_deg.max())      if len(user_deg)     > 0 else 0.0,
        'avg_merchant_degree':  float(merchant_deg.mean()) if len(merchant_deg) > 0 else 0.0,
        'max_merchant_degree':  float(merchant_deg.max())  if len(merchant_deg) > 0 else 0.0,
        'n_components':         int(n_comp),
        'largest_cc_size':      largest_cc,
        'largest_cc_frac':      float(largest_cc / (n_u + n_m)),
    }


# ------------------------------------------------------------------ #
#  Degree distribution plots                                           #
# ------------------------------------------------------------------ #
def degree_distribution_plot(B: sp.csr_matrix, name: str, out_dir: str = PLOT_DIR) -> None:
    """
    Two PNGs:
      {name}_degree_hist.png    — user/merchant degree histograms (log y)
      {name}_degree_loglog.png  — log-log rank vs degree scatter
    """
    B_bin = B.copy()
    B_bin.data = np.ones_like(B_bin.data)
    user_deg     = np.array(B_bin.sum(axis=1)).flatten()
    merchant_deg = np.array(B_bin.sum(axis=0)).flatten()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(user_deg, bins=50, color='steelblue', edgecolor='none')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('User Degree')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title(f'{name} — User Degree Distribution')

    axes[1].hist(merchant_deg, bins=50, color='coral', edgecolor='none')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Merchant Degree')
    axes[1].set_ylabel('Count (log)')
    axes[1].set_title(f'{name} — Merchant Degree Distribution')

    plt.tight_layout()
    hist_path = os.path.join(out_dir, f'{name}_degree_hist.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {hist_path}')

    # Log-log rank vs degree
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, deg, label, color in [
        (axes[0], user_deg,     'User',     'steelblue'),
        (axes[1], merchant_deg, 'Merchant', 'coral'),
    ]:
        sorted_deg = np.sort(deg)[::-1]
        ranks = np.arange(1, len(sorted_deg) + 1)
        pos   = sorted_deg > 0
        ax.scatter(ranks[pos], sorted_deg[pos], s=2, alpha=0.5, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Rank (log)')
        ax.set_ylabel('Degree (log)')
        ax.set_title(f'{name} — {label} Degree (log-log)')

    plt.tight_layout()
    loglog_path = os.path.join(out_dir, f'{name}_degree_loglog.png')
    plt.savefig(loglog_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {loglog_path}')
