"""
Script 2: Generate LaTeX tables + Markdown report.
Reads:
  - analysis/artifacts/raw_stats.json + graph_stats.json (dataset stats)
  - Results1/*_seed42.json                                (SelfGNN variants)
  - Results_baselines/*_seed42.json                       (baseline models)

Writes:
  - analysis/dataset_stats_table.tex   (formerly comparison_table.tex)
  - analysis/comparison_table.tex       (NEW: SelfGNN+baselines x 3 datasets)
  - analysis/segment_table.tex          (NEW: low/mid/high per best variant)
  - analysis/comparison_report.md       (refreshed with all three sections)

Runtime: < 5 seconds.
"""
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))
from utils import ROOT, load_artifact


def fmt_int(x) -> str:
    if x is None:
        return '---'
    return f'{int(x):,}'


def fmt_flt(x, d=4) -> str:
    if x is None:
        return '---'
    return f'{float(x):.{d}f}'


def fmt_pct(x, d=2) -> str:
    if x is None:
        return '---'
    return f'{float(x)*100:.{d}f}\\%'


# ------------------------------------------------------------------ #
#  LaTeX table                                                         #
# ------------------------------------------------------------------ #
def render_latex_table(raw: dict, graph: dict) -> str:
    y   = raw['yelp']
    fi  = raw['finance']
    sy  = raw['synthetic']
    yg  = graph['yelp']
    fig = graph['finance']
    syg = graph['synthetic']

    ya  = yg['after_kcore']
    fia = fig['after_kcore']
    sya = syg['after_kcore']
    y_gt = yg['ground_truth_kcore']

    yf  = yg['uniform_features']
    fif = fig['uniform_features']
    syf = syg['uniform_features']

    y_time  = f"{y['temporal_range']['min']}--{y['temporal_range']['max']}"
    fi_time = f"{fi['temporal_range']['min']}--{fi['temporal_range']['max']}"
    sy_time = f"{sy['temporal_range']['min']}--{sy['temporal_range']['max']}"

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Dataset comparison for SelfGNN merchant recommendation.')
    lines.append(r'         Graph metrics on 500K-edge samples; $k$-core at $k=5$.')
    lines.append(r'         All datasets share a uniform 4-D user + 4-D merchant feature schema.}')
    lines.append(r'\label{tab:dataset_comparison}')
    lines.append(r'\begin{tabular}{l r r r}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Property} & \textbf{Yelp} & \textbf{Finance} & \textbf{Synthetic} \\')
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{4}{l}{\textit{Dataset Statistics}} \\[2pt]')

    def row(label, yv, fv, sv):
        lines.append(f'{label} & {yv} & {fv} & {sv} \\\\')

    row(r'Users $|\mathcal{U}|$',
        fmt_int(y['n_users']),  fmt_int(fi['n_users']),  fmt_int(sy['n_users']))
    row(r'Merchants $|\mathcal{M}|$',
        fmt_int(y['n_merchants']), fmt_int(fi['n_merchants']), fmt_int(sy['n_merchants']))
    row(r'Interactions $|\mathcal{E}|$',
        fmt_int(y['n_interactions']), fmt_int(fi['n_interactions']), fmt_int(sy['n_interactions']))
    row('User-user social edges',
        fmt_int(y['n_social_edges']), '---', '---')
    row('Temporal range',   y_time, fi_time, sy_time)
    row(r'Interaction signal',
        r'Stars (1--5)', r'Amount (USD)', r'Amount (units)')
    row(r'Edge weight',
        r'stars$/5$', r'$\sigma(\log(1{+}|\$|)/p_{75})$', r'$\sigma(\log(1{+}a)/p_{75})$')

    lines.append(r'\midrule')
    lines.append(r'\multicolumn{4}{l}{\textit{Graph Properties after $k$-core ($k=5$)}} \\[2pt]')

    row(r'$|\mathcal{U}|_k$ full dataset',
        fmt_int(y_gt['n_users']), '---', '---')
    row(r'$|\mathcal{M}|_k$ full dataset',
        fmt_int(y_gt['n_merchants']), '---', '---')
    row(r'$|\mathcal{U}|_k$ (500K sample)',
        fmt_int(ya['n_users']),  fmt_int(fia['n_users']),  fmt_int(sya['n_users']))
    row(r'$|\mathcal{M}|_k$ (500K sample)',
        fmt_int(ya['n_merchants']),  fmt_int(fia['n_merchants']),  fmt_int(sya['n_merchants']))
    row(r'$|\mathcal{E}|_k$ (500K sample)',
        fmt_int(ya['n_edges']),  fmt_int(fia['n_edges']),  fmt_int(sya['n_edges']))
    row('Density',
        fmt_flt(ya['density'], 6), fmt_flt(fia['density'], 6), fmt_flt(sya['density'], 6))
    row('Sparsity',
        fmt_flt(ya['sparsity'], 4), fmt_flt(fia['sparsity'], 4), fmt_flt(sya['sparsity'], 4))
    row(r'Avg.\ user degree',
        fmt_flt(ya['avg_user_degree'], 1),
        fmt_flt(fia['avg_user_degree'], 1),
        fmt_flt(sya['avg_user_degree'], 1))
    row(r'Avg.\ merchant degree',
        fmt_flt(ya['avg_merchant_degree'], 1),
        fmt_flt(fia['avg_merchant_degree'], 1),
        fmt_flt(sya['avg_merchant_degree'], 1))
    row('Connected components',
        fmt_int(ya['n_components']),
        fmt_int(fia['n_components']),
        fmt_int(sya['n_components']))
    row(r'Largest CC (\% nodes)',
        fmt_pct(ya['largest_cc_frac']),
        fmt_pct(fia['largest_cc_frac']),
        fmt_pct(sya['largest_cc_frac']))
    row(r'Temporal slices $T$',
        str(yg['temporal_slices']),
        str(fig['temporal_slices']),
        str(syg['temporal_slices']))

    lines.append(r'\midrule')
    lines.append(r'\multicolumn{4}{l}{\textit{Uniform Feature Schema ($d_u{=}4$, $d_m{=}4$, $d_e{=}1$)}} \\[2pt]')

    y_u_cnt  = yf['user_features']['stats']['interaction_count']
    fi_u_cnt = fif['user_features']['stats']['interaction_count']
    sy_u_cnt = syf['user_features']['stats']['interaction_count']
    row(r'$f_u^{(1)}$ interaction count (mean)',
        fmt_flt(y_u_cnt['mean'], 1),
        fmt_flt(fi_u_cnt['mean'], 1),
        fmt_flt(sy_u_cnt['mean'], 1))

    y_u_avg  = yf['user_features']['stats']['avg_interaction_value']
    fi_u_avg = fif['user_features']['stats']['avg_interaction_value']
    sy_u_avg = syf['user_features']['stats']['avg_interaction_value']
    row(r'$f_u^{(2)}$ avg.\ interaction value (mean)',
        fmt_flt(y_u_avg['mean'], 3),
        fmt_flt(fi_u_avg['mean'], 3),
        fmt_flt(sy_u_avg['mean'], 3))

    y_u_span  = yf['user_features']['stats']['activity_span_days']
    fi_u_span = fif['user_features']['stats']['activity_span_days']
    sy_u_span = syf['user_features']['stats']['activity_span_days']
    row(r'$f_u^{(4)}$ activity span (mean days)',
        fmt_flt(y_u_span['mean'], 1),
        fmt_flt(fi_u_span['mean'], 1),
        fmt_flt(sy_u_span['mean'], 1))

    row('Explicit feedback',        r'Yes (stars)', r'No (amount)', r'No (amount)')
    row('Long-tail distribution',   r'Yes',          r'Yes',          r'\textbf{No}')
    row('Social graph available',   r'Yes',          r'No',           r'No')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ #
#  Markdown report                                                     #
# ------------------------------------------------------------------ #
def render_markdown_report(raw: dict, graph: dict) -> str:
    y   = raw['yelp']
    fi  = raw['finance']
    sy  = raw['synthetic']
    yg  = graph['yelp']
    fig = graph['finance']
    syg = graph['synthetic']
    ya  = yg['after_kcore']
    fia = fig['after_kcore']
    sya = syg['after_kcore']
    y_gt = yg['ground_truth_kcore']
    yf  = yg['uniform_features']
    fif = fig['uniform_features']
    syf = syg['uniform_features']

    lines = []
    lines.append('# Dataset Comparison Report — Merchant Recommendation (SelfGNN)')
    lines.append('')
    lines.append('All 3 datasets use **identical 4D user + 4D merchant + 1D edge feature schema** '
                 'for direct model comparison.')
    lines.append('')

    lines.append('## Uniform Feature Schema')
    lines.append('')
    lines.append('| Dim | User feature | Merchant feature |')
    lines.append('|-----|-------------|-----------------|')
    lines.append('| 1 | `interaction_count` | `interaction_count` |')
    lines.append('| 2 | `avg_interaction_value` (normalized 0–1) | `avg_interaction_value` (normalized 0–1) |')
    lines.append('| 3 | `unique_merchant_count` | `unique_user_count` |')
    lines.append('| 4 | `activity_span_days` | `category_id` (int-encoded) |')
    lines.append('')
    lines.append('**Edge weight normalization:**')
    lines.append('- Yelp: `stars / 5.0` → [0.2, 1.0]')
    lines.append('- Finance: `sigmoid(log1p(|amount|) / p75)` → (0, 1)')
    lines.append('- Synthetic: `sigmoid(log1p(amount) / p75)` → (0, 1)')
    lines.append('')

    lines.append('## Dataset Statistics')
    lines.append('')
    lines.append('| Statistic | Yelp | Finance | Synthetic |')
    lines.append('|-----------|------|---------|-----------|')

    def r(label, yv, fv, sv):
        lines.append(f'| {label} | {yv} | {fv} | {sv} |')

    r('Users |U|',        fmt_int(y['n_users']),        fmt_int(fi['n_users']),       fmt_int(sy['n_users']))
    r('Merchants |M|',   fmt_int(y['n_merchants']),    fmt_int(fi['n_merchants']),   fmt_int(sy['n_merchants']))
    r('Interactions |E|', fmt_int(y['n_interactions']), fmt_int(fi['n_interactions']), fmt_int(sy['n_interactions']))
    r('Social edges',     fmt_int(y['n_social_edges']), '---', '---')
    r('Temporal range',
      f"{y['temporal_range']['min']}–{y['temporal_range']['max']}",
      f"{fi['temporal_range']['min']}–{fi['temporal_range']['max']}",
      f"{sy['temporal_range']['min']}–{sy['temporal_range']['max']}")

    lines.append('')
    lines.append('## Graph Properties (k-core k=5, 500K-edge sample)')
    lines.append('')
    lines.append('| Metric | Yelp | Finance | Synthetic |')
    lines.append('|--------|------|---------|-----------|')

    r('|U|_k full dataset',   fmt_int(y_gt['n_users']),      '---',  '---')
    r('|M|_k full dataset',   fmt_int(y_gt['n_merchants']),  '---',  '---')
    r('|U|_k (sample)',       fmt_int(ya['n_users']),    fmt_int(fia['n_users']),    fmt_int(sya['n_users']))
    r('|M|_k (sample)',       fmt_int(ya['n_merchants']), fmt_int(fia['n_merchants']), fmt_int(sya['n_merchants']))
    r('|E|_k (sample)',       fmt_int(ya['n_edges']),    fmt_int(fia['n_edges']),    fmt_int(sya['n_edges']))
    r('Density',              fmt_flt(ya['density'], 6), fmt_flt(fia['density'], 6), fmt_flt(sya['density'], 6))
    r('Sparsity',             fmt_flt(ya['sparsity'], 4), fmt_flt(fia['sparsity'], 4), fmt_flt(sya['sparsity'], 4))
    r('Avg user degree',      fmt_flt(ya['avg_user_degree'], 2), fmt_flt(fia['avg_user_degree'], 2), fmt_flt(sya['avg_user_degree'], 2))
    r('Avg merchant degree',  fmt_flt(ya['avg_merchant_degree'], 2), fmt_flt(fia['avg_merchant_degree'], 2), fmt_flt(sya['avg_merchant_degree'], 2))
    r('Components',           fmt_int(ya['n_components']), fmt_int(fia['n_components']), fmt_int(sya['n_components']))
    r('Largest CC (%)',       f"{ya['largest_cc_frac']*100:.2f}%", f"{fia['largest_cc_frac']*100:.2f}%", f"{sya['largest_cc_frac']*100:.2f}%")
    r('Temporal slices T',    str(yg['temporal_slices']), str(fig['temporal_slices']), str(syg['temporal_slices']))

    lines.append('')
    lines.append('## Uniform Feature Statistics (after k-core, 500K sample)')
    lines.append('')
    lines.append('| Feature | Yelp mean±std | Finance mean±std | Synthetic mean±std |')
    lines.append('|---------|--------------|-----------------|-------------------|')

    def feat_row(label, key, side):
        sy_s  = yf[side]['stats'][key]
        fi_s  = fif[side]['stats'][key]
        syn_s = syf[side]['stats'][key]
        lines.append(
            f'| {label} '
            f'| {sy_s["mean"]:.3f}±{sy_s["std"]:.3f} '
            f'| {fi_s["mean"]:.3f}±{fi_s["std"]:.3f} '
            f'| {syn_s["mean"]:.3f}±{syn_s["std"]:.3f} |'
        )

    lines.append('**User features:**')
    feat_row('f_u[1] interaction_count',      'interaction_count',      'user_features')
    feat_row('f_u[2] avg_interaction_value',  'avg_interaction_value',  'user_features')
    feat_row('f_u[3] unique_merchant_count',  'unique_merchant_count',  'user_features')
    feat_row('f_u[4] activity_span_days',     'activity_span_days',     'user_features')
    lines.append('')
    lines.append('**Merchant features:**')
    feat_row('f_m[1] interaction_count',     'interaction_count',     'merchant_features')
    feat_row('f_m[2] avg_interaction_value', 'avg_interaction_value', 'merchant_features')
    feat_row('f_m[3] unique_user_count',     'unique_user_count',     'merchant_features')
    feat_row('f_m[4] category_id',           'category_id',           'merchant_features')

    lines.append('')
    lines.append('## Recommendation Suitability')
    lines.append('')
    lines.append('| Property | Yelp | Finance | Synthetic |')
    lines.append('|----------|------|---------|-----------|')
    lines.append('| Explicit feedback | Yes (stars) | No (implicit) | No (implicit) |')
    lines.append('| Long-tail distribution | Yes | Yes | **No** (only 48–49 merchants) |')
    lines.append('| Social graph | Yes | No | No |')
    lines.append('| d_u user features | 4 | 4 | 4 |')
    lines.append('| d_m merchant features | 4 | 4 | 4 |')
    lines.append('| d_e edge features | 1 | 1 | 1 |')

    lines.append('')
    lines.append('## Recommendations')
    lines.append('')
    lines.append('### Primary benchmark: **Yelp**')
    lines.append(f'- {fmt_int(y_gt["n_users"])} users × {fmt_int(y_gt["n_merchants"])} merchants (full k-core)')
    lines.append(f'- Explicit star ratings → cleanest recommendation ground truth')
    lines.append(f'- {yg["temporal_slices"]} temporal slices for SelfGNN self-supervised learning')
    lines.append('')
    lines.append('### Implicit feedback benchmark: **Finance**')
    lines.append(f'- {fmt_int(fi["n_users"])} users × {fmt_int(fi["n_merchants"])} merchants — sparse realistic graph')
    lines.append(f'- {fig["temporal_slices"]} annual time slices (2010–2019)')
    lines.append(f'- Transaction amount as implicit preference proxy')
    lines.append('')
    lines.append('### Controlled experiments only: **Synthetic**')
    lines.append(f'- {fmt_int(sy["n_merchants"])} merchants → density {sya["density"]*100:.1f}% — near-complete graph')
    lines.append(f'- GNN message passing provides minimal signal in dense graphs')
    lines.append(f'- Useful for scalability / ablation studies only')
    lines.append('')
    lines.append('---')
    lines.append('*Generated by `analysis/02_generate_outputs.py`*')

    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print('Loading artifacts...')
    raw   = load_artifact('raw_stats')
    graph = load_artifact('graph_stats')

    print('Generating dataset-stats LaTeX table...')
    tex = render_latex_table(raw, graph)
    dataset_tex_path = os.path.join(ROOT, 'analysis', 'dataset_stats_table.tex')
    with open(dataset_tex_path, 'w') as f:
        f.write(tex)
    print(f'  Saved: {dataset_tex_path}')

    # ------------------------------------------------------------------ #
    #  Model-results tables (comparison + segments)                       #
    # ------------------------------------------------------------------ #
    print('Generating model comparison + segment tables...')
    results = _load_model_results()
    comp_tex = _render_comparison_table(results)
    seg_tex  = _render_segment_table(results)

    comp_path = os.path.join(ROOT, 'analysis', 'comparison_table.tex')
    with open(comp_path, 'w') as f:
        f.write(comp_tex)
    print(f'  Saved: {comp_path}')

    seg_path = os.path.join(ROOT, 'analysis', 'segment_table.tex')
    with open(seg_path, 'w') as f:
        f.write(seg_tex)
    print(f'  Saved: {seg_path}')

    print('Generating Markdown report...')
    md = render_markdown_report(raw, graph)
    md += '\n\n' + _render_results_markdown(results)
    md_path = os.path.join(ROOT, 'analysis', 'comparison_report.md')
    with open(md_path, 'w') as f:
        f.write(md)
    print(f'  Saved: {md_path}')

    print('\nDone.')


# ------------------------------------------------------------------ #
#  Model results — ingest + tables                                    #
# ------------------------------------------------------------------ #

# The five baselines + four SelfGNN variants in display order.
MODEL_TAGS = [
    ('popularity',        'Popularity',       'baseline'),
    ('bprmf',             'BPRMF',            'baseline'),
    ('lightgcn',          'LightGCN',         'baseline'),
    ('sasrec',            'SASRec',           'baseline'),
    ('bert4rec',          'BERT4Rec',         'baseline'),
    ('base',              'SelfGNN-Base',     'selfgnn'),
    ('node',              'SelfGNN + Node',   'selfgnn'),
    ('edge',              'SelfGNN + Edge',   'selfgnn'),
    ('node_edge',         'SelfGNN + Both',   'selfgnn'),
]

# Dataset display name → file prefix used in Results/ paths.
DATASETS = [
    ('yelp',      'yelp_merchant',      'Yelp'),
    ('finance',   'finance_merchant',   'Finance'),
    ('synthetic', 'synthetic_merchant', 'Synthetic'),
]


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_model_results(seed: int = 42) -> dict:
    """Return {dataset_key: {model_key: {overall, segments}}} for seed."""
    out: dict = {d_key: {} for d_key, _, _ in DATASETS}
    for d_key, d_prefix, _ in DATASETS:
        for m_key, _, kind in MODEL_TAGS:
            sub  = 'Results1' if kind == 'selfgnn' else 'Results_baselines'
            path = os.path.join(ROOT, sub, f'{d_prefix}_{m_key}_seed{seed}.json')
            js   = _read_json(path)
            if not js:
                continue
            out[d_key][m_key] = {
                'overall':  js.get('test_results', {}) or {},
                'segments': js.get('test_segments', {}) or {},
                'epoch':    js.get('best_epoch'),
            }
    return out


def _render_comparison_table(results: dict) -> str:
    """LaTeX: one row per method, two columns per dataset (HR@10, NDCG@10)."""
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Merchant-recommendation test performance. Seed 42, '
        r'balanced low/mid/high test groups, 999 sampled negatives. '
        r'Best per column in \textbf{bold}; second-best \underline{underlined}.}',
        r'\label{tab:results}',
        r'\begin{tabular}{l cc cc cc}',
        r'\toprule',
        r'\textbf{Method} ' + ''.join(
            f'& \\multicolumn{{2}}{{c}}{{\\textbf{{{dname}}}}} '
            for _, _, dname in DATASETS
        ) + r'\\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}',
        r' & HR@10 & NDCG@10 ' * 1
        + r'& HR@10 & NDCG@10 ' * 1
        + r'& HR@10 & NDCG@10 \\',
        r'\midrule',
    ]

    # Find per-column best / 2nd-best for bold / underline.
    best = {}
    for d_key, _, _ in DATASETS:
        for metric in ('HR@10', 'NDCG@10'):
            vals = [
                (m_key, results[d_key][m_key]['overall'].get(metric))
                for m_key, _, _ in MODEL_TAGS
                if m_key in results[d_key]
                and results[d_key][m_key]['overall'].get(metric) is not None
            ]
            vals.sort(key=lambda x: x[1], reverse=True)
            best[(d_key, metric)] = {
                'first':  vals[0][0] if len(vals) >= 1 else None,
                'second': vals[1][0] if len(vals) >= 2 else None,
            }

    def _cell(val, fmt_str, is_first, is_second):
        if val is None:
            return '---'
        s = fmt_str.format(val)
        if is_first:
            return r'\textbf{' + s + '}'
        if is_second:
            return r'\underline{' + s + '}'
        return s

    prev_kind = None
    for m_key, m_label, kind in MODEL_TAGS:
        if prev_kind is not None and kind != prev_kind:
            lines.append(r'\midrule')
        prev_kind = kind
        row = [m_label]
        for d_key, _, _ in DATASETS:
            rec = results.get(d_key, {}).get(m_key)
            if rec is None:
                row.extend(['---', '---'])
                continue
            for metric, fmt_str in (('HR@10', '{:.4f}'), ('NDCG@10', '{:.4f}')):
                v   = rec['overall'].get(metric)
                b   = best[(d_key, metric)]
                row.append(_cell(v, fmt_str, m_key == b['first'], m_key == b['second']))
        lines.append(' & '.join(row) + r' \\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    return '\n'.join(lines)


def _render_segment_table(results: dict) -> str:
    """LaTeX: per dataset, pick the SelfGNN variant with best overall NDCG@10
    and display low / mid / high HR@10 and NDCG@10."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Per-segment test performance of the strongest SelfGNN '
        r'variant on each dataset. Low/mid/high are size-balanced '
        r'tercile groups (3{,}333 users each) of test users ranked by '
        r'training-event count.}',
        r'\label{tab:segments}',
        r'\begin{tabular}{l l cc cc cc}',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{Variant} '
        r'& \multicolumn{2}{c}{\textbf{Low}} '
        r'& \multicolumn{2}{c}{\textbf{Mid}} '
        r'& \multicolumn{2}{c}{\textbf{High}} \\',
        r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}',
        r' & & HR@10 & NDCG@10 & HR@10 & NDCG@10 & HR@10 & NDCG@10 \\',
        r'\midrule',
    ]

    selfgnn_keys = [k for k, _, kind in MODEL_TAGS if kind == 'selfgnn']
    label_lookup = {k: label for k, label, _ in MODEL_TAGS}

    for d_key, _, d_label in DATASETS:
        ds = results.get(d_key, {})
        best_key, best_ndcg = None, -1.0
        for k in selfgnn_keys:
            if k not in ds:
                continue
            v = ds[k]['overall'].get('NDCG@10')
            if v is not None and v > best_ndcg:
                best_ndcg, best_key = v, k
        if best_key is None:
            lines.append(f'{d_label} & --- & --- & --- & --- & --- & --- & --- \\\\')
            continue
        seg = ds[best_key].get('segments', {}) or {}
        row = [d_label, label_lookup[best_key]]
        for seg_name in ('low', 'mid', 'high'):
            m = seg.get(seg_name, {})
            hr   = m.get('HR@10')
            ndcg = m.get('NDCG@10')
            row.append(f'{hr:.4f}'   if hr   is not None else '---')
            row.append(f'{ndcg:.4f}' if ndcg is not None else '---')
        lines.append(' & '.join(row) + r' \\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def _render_results_markdown(results: dict) -> str:
    """Markdown mirror of the model tables, appended to comparison_report.md."""
    lines = ['## Model Comparison (seed 42, test set)', '']
    header = '| Method | ' + ' | '.join(
        f'{d} HR@10 | {d} NDCG@10' for _, _, d in DATASETS
    ) + ' |'
    sep    = '|--------|' + '-----------|' * (2 * len(DATASETS))
    lines.append(header)
    lines.append(sep)
    prev_kind = None
    for m_key, m_label, kind in MODEL_TAGS:
        if prev_kind is not None and kind != prev_kind:
            lines.append('|' + ' --- |' * (1 + 2 * len(DATASETS)))
        prev_kind = kind
        row = [f'| {m_label}']
        for d_key, _, _ in DATASETS:
            rec = results.get(d_key, {}).get(m_key)
            if rec is None:
                row.extend([' ---', ' ---'])
                continue
            hr   = rec['overall'].get('HR@10')
            ndcg = rec['overall'].get('NDCG@10')
            row.append(f" {hr:.4f}"   if hr   is not None else ' ---')
            row.append(f" {ndcg:.4f}" if ndcg is not None else ' ---')
        lines.append(' |'.join(row) + ' |')

    lines.append('')
    lines.append('## Segment Metrics — Best SelfGNN Variant per Dataset')
    lines.append('')
    lines.append('| Dataset | Variant | Low HR@10 | Low NDCG@10 | '
                 'Mid HR@10 | Mid NDCG@10 | High HR@10 | High NDCG@10 |')
    lines.append('|---------|---------|-----------|-------------|'
                 '-----------|-------------|------------|--------------|')
    selfgnn_keys = [k for k, _, kind in MODEL_TAGS if kind == 'selfgnn']
    label_lookup = {k: label for k, label, _ in MODEL_TAGS}
    for d_key, _, d_label in DATASETS:
        ds = results.get(d_key, {})
        best_key, best_ndcg = None, -1.0
        for k in selfgnn_keys:
            if k not in ds:
                continue
            v = ds[k]['overall'].get('NDCG@10')
            if v is not None and v > best_ndcg:
                best_ndcg, best_key = v, k
        if best_key is None:
            lines.append(f'| {d_label} | --- | --- | --- | --- | --- | --- | --- |')
            continue
        seg = ds[best_key].get('segments', {}) or {}
        cells = [d_label, label_lookup[best_key]]
        for s in ('low', 'mid', 'high'):
            m = seg.get(s, {})
            hr, nd = m.get('HR@10'), m.get('NDCG@10')
            cells.append(f'{hr:.4f}'  if hr is not None else '---')
            cells.append(f'{nd:.4f}'  if nd is not None else '---')
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
