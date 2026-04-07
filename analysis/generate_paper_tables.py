"""
Generate LaTeX tables for the paper from the multi-seed comparison summary.

Reads:
    Results/comparison_summary.json   (written by analysis/compare_results.py)
    Results/dataset_statistics.json   (optional; uses defaults if missing)

Writes:
    paper_figures/table_dataset.tex
    paper_figures/table_configs.tex
    paper_figures/table_results.tex
    paper_figures/table_improvements.tex
    paper_figures/table_hyperparams.tex
"""
import os
import json

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(_root, 'Results')
FIGURES_DIR = os.path.join(_root, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
CONFIG_NAMES = [
    'T1 Baseline',
    'T2 Edge',
    'T3 Node',
    'T4 Both (dup)',
    'T4 Both (nodup)',
]


def load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def best_in_col(summary, metric):
    best = -1.0
    for name in CONFIG_NAMES:
        if name in summary:
            val = summary[name].get('test_mean', {}).get(metric, 0.0)
            if val > best:
                best = val
    return best


# ------------------------------------------------------------------ #
#  table_dataset                                                       #
# ------------------------------------------------------------------ #
def table_dataset(stats):
    users = stats.get('users', 'N/A')
    merchants = stats.get('merchants', 'N/A')
    train_events = stats.get('total_train_events', 'N/A')
    density = stats.get('density', 'N/A')
    ud = stats.get('user_degree', {})
    md = stats.get('merchant_degree', {})
    test_users = stats.get('test_users', 'N/A')
    val_users = stats.get('val_users', 'N/A')
    intervals = stats.get('time_intervals', 5)

    def fmt(v, d=4):
        if isinstance(v, float):
            return f'{v:.{d}f}'
        return str(v)

    rows = [
        ('Хэрэглэгч (Users)',
         fmt(users, 0) if isinstance(users, (int, float)) else users),
        ('Борлуулагч (Merchants)',
         fmt(merchants, 0) if isinstance(merchants, (int, float)) else merchants),
        ('Сургалтын харилцан үйлдэл (Train interactions)',
         f'{train_events:,}' if isinstance(train_events, (int, float)) else train_events),
        ('Нягтрал (Density)',
         f'{density:.4f}\\%' if isinstance(density, float) else density),
        ('Хэрэглэгч/у дундаж зэрэг (Avg merchants/user)',
         fmt(ud.get('mean', 'N/A'))),
        ('Борлуулагч/у дундаж зэрэг (Avg users/merchant)',
         fmt(md.get('mean', 'N/A'))),
        ('Тест хэрэглэгч (Test users)', str(test_users)),
        ('Баталгаажуулалт хэрэглэгч (Val users)', str(val_users)),
        ('Цагийн интервал (T)', str(intervals)),
    ]

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Yelp-Merchant өгөгдлийн олонлогийн статистик}',
        r'\label{tab:dataset}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Үзүүлэлт} & \textbf{Утга} \\',
        r'\midrule',
    ]
    for label, val in rows:
        lines.append(f'{label} & {val} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  table_configs                                                       #
# ------------------------------------------------------------------ #
def table_configs():
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Туршилтын 5 тохиргоо}',
        r'\label{tab:configs}',
        r'\begin{tabular}{llll}',
        r'\toprule',
        r'\textbf{Тохиргоо} & \textbf{Граф} & \textbf{Оройн шинж} & \textbf{Ирмэгийн шинж} \\',
        r'\midrule',
        r'T1 Baseline      & Хоёртын & ---                                   & ---                                       \\',
        r'T2 Edge          & Жинт   & ---                                   & Үнэлгээ (log-sigmoid + нормчлол)        \\',
        r'T3 Node          & Хоёртын & MLP проекц (нэмэх)                   & ---                                       \\',
        r'T4 Both (dup)    & Жинт   & MLP проекц (одтой шинж хадгалагдсан) & Үнэлгээ (log-sigmoid + нормчлол)        \\',
        r'T4 Both (nodup)  & Жинт   & MLP проекц (одтой шинж тэглэгдсэн)   & Үнэлгээ (log-sigmoid + нормчлол)        \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  table_results                                                       #
# ------------------------------------------------------------------ #
def table_results(summary):
    r"""Cell format: \textbf{0.3482} $\pm$ 0.0021 ($\uparrow$+1.3\%)"""
    best = {m: best_in_col(summary, m) for m in METRICS}

    def fmt_cell(name, metric):
        if name not in summary:
            return '---'
        entry = summary[name]
        mu = entry.get('test_mean', {}).get(metric, 0.0)
        sd = entry.get('test_std', {}).get(metric, 0.0)
        imp_dict = entry.get('improvement_over_baseline_mean') or {}
        imp = imp_dict.get(metric)
        is_best = best.get(metric, -1) > 0 and abs(mu - best[metric]) < 1e-9
        body = f'{mu:.4f}'
        if is_best:
            body = r'\textbf{' + body + '}'
        cell = f'{body} $\\pm$ {sd:.4f}'
        if imp is not None and name != 'T1 Baseline':
            arrow = r'\uparrow' if imp >= 0 else r'\downarrow'
            sign = '+' if imp >= 0 else ''
            cell += f' (${arrow}${sign}{imp:.2f}\\%)'
        return cell

    header_cols = ' & '.join([r'\textbf{' + m + '}' for m in METRICS])
    lines = [
        r'\begin{table*}[h]',
        r'\centering',
        r'\caption{5 тохиргооны туршилтын үр дүнгийн харьцуулалт (дундаж $\pm$ стандарт хазайлт; гурван үр)}',
        r'\label{tab:results}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Тохиргоо} & ' + header_cols + r' \\',
        r'\midrule',
    ]
    for name in CONFIG_NAMES:
        cells = ' & '.join([fmt_cell(name, m) for m in METRICS])
        lines.append(f'{name} & {cells} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  table_improvements                                                  #
# ------------------------------------------------------------------ #
def table_improvements(summary):
    """% improvement of each non-baseline config over T1 Baseline mean."""
    def fmt(name, metric):
        if name not in summary:
            return '---'
        entry = summary[name]
        imp_m = (entry.get('improvement_over_baseline_mean') or {}).get(metric)
        imp_s = (entry.get('improvement_over_baseline_std') or {}).get(metric)
        if imp_m is None:
            return '---'
        sign = '+' if imp_m >= 0 else ''
        s_part = f' $\\pm$ {imp_s:.2f}' if imp_s is not None else ''
        return f'{sign}{imp_m:.2f}{s_part}\\%'

    header_cols = ' & '.join([r'\textbf{' + m + '}' for m in METRICS])
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Суурь загвартай харьцуулсан гүйцэтгэлийн өөрчлөлт (\%)}',
        r'\label{tab:improvements}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Тохиргоо} & ' + header_cols + r' \\',
        r'\midrule',
    ]
    for name in CONFIG_NAMES[1:]:
        cells = ' & '.join([fmt(name, m) for m in METRICS])
        lines.append(f'{name} & {cells} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  table_hyperparams                                                   #
# ------------------------------------------------------------------ #
def table_hyperparams():
    rows = [
        ('Embedding dim ($d$)', '64'),
        ('GNN давхарга ($L$)', '3'),
        ('Анхаарлын давхарга', '2'),
        ('Batch хэмжээ', '512'),
        ('Сургалтын хурд ($\\eta$)', '$10^{-3}$'),
        ('L2 регуляризаци ($\\lambda$)', '$10^{-2}$'),
        ('SAL жин ($\\lambda_{ssl}$)', '$10^{-7}$'),
        ('Ирмэгийн dropout хурд', '0.5'),
        ('LSTM далд хэмжээ', '64'),
        ('Анхаарлын толгой', '16'),
        ('Дарааллын урт', '200'),
        ('Цагийн интервал ($T$)', '5'),
        ('Сургалтын урын тоо (epoch)', '150'),
        ('Эрт зогсоолт (patience)', '20'),
        ('Үрүүд (seeds)', '$\\{48, 2048, 100\\}$'),
    ]
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Сургалтын гиперпараметрүүд}',
        r'\label{tab:hyperparams}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Параметр} & \textbf{Утга} \\',
        r'\midrule',
    ]
    for param, val in rows:
        lines.append(f'{param} & {val} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    summary = load_json(os.path.join(RESULTS_DIR, 'comparison_summary.json'))
    stats = load_json(os.path.join(RESULTS_DIR, 'dataset_statistics.json'))

    if not summary:
        print('WARNING: comparison_summary.json not found or empty. '
              'Run analysis/compare_results.py first.')

    tables = {
        'table_dataset.tex':       table_dataset(stats),
        'table_configs.tex':       table_configs(),
        'table_results.tex':       table_results(summary),
        'table_improvements.tex':  table_improvements(summary),
        'table_hyperparams.tex':   table_hyperparams(),
    }

    for fname, tex in tables.items():
        path = os.path.join(FIGURES_DIR, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tex)
        print(f'\n=== {fname} ===')
        print(tex)

    print(f'\nAll tables saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
