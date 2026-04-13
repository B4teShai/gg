"""
Script 2: Generate LaTeX table + Markdown report.
Reads analysis/artifacts/raw_stats.json + graph_stats.json.
Writes analysis/comparison_table.tex + analysis/comparison_report.md.
Runtime: < 5 seconds.
"""
import os
import sys

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

    print('Generating LaTeX table...')
    tex = render_latex_table(raw, graph)
    tex_path = os.path.join(ROOT, 'analysis', 'comparison_table.tex')
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'  Saved: {tex_path}')

    print('Generating Markdown report...')
    md = render_markdown_report(raw, graph)
    md_path = os.path.join(ROOT, 'analysis', 'comparison_report.md')
    with open(md_path, 'w') as f:
        f.write(md)
    print(f'  Saved: {md_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
