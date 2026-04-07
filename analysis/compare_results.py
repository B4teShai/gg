"""
Multi-seed results comparison and figure generation for the SelfGNN paper.

Discovers all per-run JSON files in Results/ matching the pattern
    {tag}_s{seed}.json
where tag ∈ {t1_base, t2_edge, t3_node, t4_dup, t4_nodup}, groups them by
config, and computes mean ± std across seeds for every test metric and every
epoch of the training history.

Outputs:
    Results/comparison_summary.json
    paper-tex/fig_convergence.png        (2×2, all 4 metrics, used by the .tex)
    paper_figures/convergence.pdf        (PDF version of the same plot)
    paper_figures/results_bar_chart.pdf  (5 configs × 4 metrics, error bars)
    paper_figures/improvement_chart.pdf  (% improvement vs baseline, error bars)
    paper_figures/loss_curve.pdf         (training loss vs epoch, error bands)
    paper_figures/seed_variance.pdf      (per-seed scatter)

Usage:
    python analysis/compare_results.py
"""
import os
import re
import glob
import json
import numpy as np

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(_root, 'Results')
FIGURES_DIR = os.path.join(_root, 'paper_figures')
PAPER_TEX_DIR = os.path.join(_root, 'paper-tex')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PAPER_TEX_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Configuration registry                                              #
# ------------------------------------------------------------------ #
# tag → (display name, color, marker)
CONFIGS = {
    't1_base':   ('T1 Baseline',     '#808080', 'o'),
    't2_edge':   ('T2 Edge',         '#FF8C00', 's'),
    't3_node':   ('T3 Node',         '#1E90FF', '^'),
    't4_dup':    ('T4 Both (dup)',   '#C71585', 'D'),
    't4_nodup':  ('T4 Both (nodup)', '#2E8B57', 'v'),
}
DISPLAY_ORDER = ['t1_base', 't2_edge', 't3_node', 't4_dup', 't4_nodup']
METRICS = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']

FNAME_RE = re.compile(r'^(t[1-4]_[a-z]+)_s(\d+)\.json$')


# ------------------------------------------------------------------ #
#  Discovery & loading                                                 #
# ------------------------------------------------------------------ #
def discover_runs():
    """Return {tag: {seed_int: result_dict}} for every recognised file."""
    runs = {tag: {} for tag in CONFIGS}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json'))):
        fname = os.path.basename(path)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        tag, seed_str = m.group(1), m.group(2)
        if tag not in CONFIGS:
            continue
        with open(path) as f:
            runs[tag][int(seed_str)] = json.load(f)
    return runs


def get_test_metrics(data):
    return data.get('test_results', data.get('final_test', {}))


# ------------------------------------------------------------------ #
#  Aggregation                                                         #
# ------------------------------------------------------------------ #
def aggregate_test(runs):
    """Returns {tag: {'n_seeds', 'test_mean', 'test_std', 'test_per_seed'}}."""
    out = {}
    for tag in DISPLAY_ORDER:
        seed_results = runs.get(tag, {})
        if not seed_results:
            continue
        per_seed = {}
        stacked = {m: [] for m in METRICS}
        for seed, data in sorted(seed_results.items()):
            test = get_test_metrics(data)
            per_seed[str(seed)] = {m: float(test.get(m, 0.0)) for m in METRICS}
            for m in METRICS:
                stacked[m].append(float(test.get(m, 0.0)))
        out[tag] = {
            'n_seeds': len(seed_results),
            'test_mean': {m: float(np.mean(stacked[m])) for m in METRICS},
            'test_std':  {m: float(np.std(stacked[m], ddof=0)) for m in METRICS},
            'test_per_seed': per_seed,
            'best_epochs': [int(d.get('best_epoch', 0))
                            for d in seed_results.values()],
        }
    return out


def aggregate_history(runs, key):
    """Stack train_history[key] across seeds for each config.

    Returns {tag: (epochs, mean_arr, std_arr)} truncated to the shortest run.
    """
    out = {}
    for tag in DISPLAY_ORDER:
        seed_results = runs.get(tag, {})
        if not seed_results:
            continue
        curves = []
        epochs_ref = None
        for data in seed_results.values():
            history = data.get('train_history', [])
            if not history:
                continue
            ep = [int(h['epoch']) for h in history]
            vals = [float(h.get(key, 0.0)) for h in history]
            curves.append((ep, vals))
            if epochs_ref is None or len(ep) < len(epochs_ref):
                epochs_ref = ep
        if not curves or epochs_ref is None:
            continue
        cut = len(epochs_ref)
        stacked = np.array([c[1][:cut] for c in curves])  # (n_seeds, cut)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=0)
        out[tag] = (np.array(epochs_ref[:cut]), mean, std)
    return out


def add_improvements(summary):
    """Compute % improvement of each config over T1 mean."""
    if 't1_base' not in summary:
        return summary
    base_mean = summary['t1_base']['test_mean']
    base_std = summary['t1_base']['test_std']
    for tag, entry in summary.items():
        if tag == 't1_base':
            entry['improvement_over_baseline_mean'] = None
            continue
        imp_mean = {}
        imp_std = {}
        for m in METRICS:
            b = base_mean.get(m, 1e-12) or 1e-12
            v_mean = entry['test_mean'].get(m, 0.0)
            v_std = entry['test_std'].get(m, 0.0)
            imp_mean[m] = (v_mean - b) / b * 100.0
            # Error propagation: rel std of (v/b - 1) ≈ sqrt((sv/v)^2 + (sb/b)^2) * |v/b|
            # We report absolute std on the percent figure.
            rel_v = v_std / max(abs(v_mean), 1e-12)
            rel_b = base_std.get(m, 0.0) / b
            imp_std[m] = abs(v_mean / b) * np.sqrt(rel_v ** 2 + rel_b ** 2) * 100.0
        entry['improvement_over_baseline_mean'] = imp_mean
        entry['improvement_over_baseline_std'] = imp_std
    return summary


# ------------------------------------------------------------------ #
#  Console reporting                                                   #
# ------------------------------------------------------------------ #
def print_table(summary):
    print('\n' + '=' * 96)
    print(f'{"Config":<18} {"n":>3}   ' +
          '   '.join(f'{m:^18}' for m in METRICS))
    print('-' * 96)
    for tag in DISPLAY_ORDER:
        if tag not in summary:
            continue
        e = summary[tag]
        cells = []
        for m in METRICS:
            mu = e['test_mean'][m]
            sd = e['test_std'][m]
            cells.append(f'{mu:.4f}±{sd:.4f}')
        print(f'{CONFIGS[tag][0]:<18} {e["n_seeds"]:>3}   ' +
              '   '.join(f'{c:^18}' for c in cells))
    print('=' * 96)


def print_improvements(summary):
    if 't1_base' not in summary:
        return
    print('\nImprovement over T1 Baseline (mean ± propagated std):')
    print('-' * 96)
    for tag in DISPLAY_ORDER[1:]:
        if tag not in summary:
            continue
        e = summary[tag]
        imp_m = e.get('improvement_over_baseline_mean') or {}
        imp_s = e.get('improvement_over_baseline_std') or {}
        cells = [f'{imp_m.get(m, 0):+6.2f}±{imp_s.get(m, 0):.2f}%' for m in METRICS]
        print(f'  {CONFIGS[tag][0]:<18} ' + '  '.join(f'{c:>16}' for c in cells))


# ------------------------------------------------------------------ #
#  Figures                                                             #
# ------------------------------------------------------------------ #
def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.family'] = 'DejaVu Serif'
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    return plt


def fig_convergence(runs):
    """2×2 panel of HR@10, HR@20, NDCG@10, NDCG@20 with shaded ±1σ bands.

    Saved to BOTH paper-tex/fig_convergence.png (referenced by the .tex)
    AND paper_figures/convergence.pdf.
    """
    plt = _setup_mpl()
    panels = [
        ('val_HR10',   'HR@10'),
        ('val_HR20',   'HR@20'),
        ('val_NDCG10', 'NDCG@10'),
        ('val_NDCG20', 'NDCG@20'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    axes_flat = axes.flatten()

    for ax, (key, label) in zip(axes_flat, panels):
        agg = aggregate_history(runs, key)
        for tag in DISPLAY_ORDER:
            if tag not in agg:
                continue
            ep, mean, std = agg[tag]
            display, color, marker = CONFIGS[tag]
            ax.plot(ep, mean, label=display, color=color, marker=marker,
                    markersize=3, linewidth=1.5)
            if (std > 0).any():
                ax.fill_between(ep, mean - std, mean + std, color=color,
                                alpha=0.18, linewidth=0)
            # Mark mean best epoch
            best_idx = int(np.argmax(mean))
            ax.plot(ep[best_idx], mean[best_idx], '*', color=color,
                    markersize=11, zorder=5,
                    markeredgecolor='black', markeredgewidth=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

    for ax in axes[1]:
        ax.set_xlabel('Epoch', fontsize=9)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(handles), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Training Convergence (mean ± 1σ across seeds)',
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    png_path = os.path.join(PAPER_TEX_DIR, 'fig_convergence.png')
    pdf_path = os.path.join(FIGURES_DIR, 'convergence.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')


def fig_results_bar(summary):
    plt = _setup_mpl()
    configs_present = [t for t in DISPLAY_ORDER if t in summary]
    if not configs_present:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n_metrics = len(METRICS)
    n_configs = len(configs_present)
    x = np.arange(n_metrics)
    width = 0.85 / n_configs
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, tag in enumerate(configs_present):
        e = summary[tag]
        means = [e['test_mean'][m] for m in METRICS]
        stds = [e['test_std'][m] for m in METRICS]
        display, color, _ = CONFIGS[tag]
        bars = ax.bar(x + offsets[i], means, width, yerr=stds, capsize=2.5,
                      label=display, color=color, alpha=0.88,
                      error_kw={'elinewidth': 0.8, 'ecolor': '#222'})
        for bar, mu, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sd + 0.003,
                    f'{mu:.3f}', ha='center', va='bottom', fontsize=6.5)

    all_vals = [summary[t]['test_mean'][m] for t in configs_present for m in METRICS]
    if all_vals:
        ymin = max(0.0, min(all_vals) - 0.04)
        ax.set_ylim(bottom=ymin)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('SelfGNN Configuration Comparison (mean ± 1σ across seeds)',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=n_configs, loc='lower center',
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'results_bar_chart.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_improvement(summary):
    plt = _setup_mpl()
    if 't1_base' not in summary:
        return
    others = [t for t in DISPLAY_ORDER[1:] if t in summary]
    if not others:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n_metrics = len(METRICS)
    n_configs = len(others)
    x = np.arange(n_metrics)
    width = 0.85 / n_configs
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, tag in enumerate(others):
        e = summary[tag]
        imp_m = e.get('improvement_over_baseline_mean') or {}
        imp_s = e.get('improvement_over_baseline_std') or {}
        means = [imp_m.get(m, 0.0) for m in METRICS]
        stds = [imp_s.get(m, 0.0) for m in METRICS]
        display, color, _ = CONFIGS[tag]
        bars = ax.bar(x + offsets[i], means, width, yerr=stds, capsize=2.5,
                      label=display, color=color, alpha=0.88,
                      error_kw={'elinewidth': 0.8, 'ecolor': '#222'})
        for bar, mu in zip(bars, means):
            offset_y = 0.10 if mu >= 0 else -0.35
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset_y,
                    f'{mu:+.1f}%', ha='center', va='bottom', fontsize=6.5)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylabel('Improvement over T1 Baseline (%)', fontsize=10)
    ax.set_title('Feature Impact (mean ± propagated std across seeds)',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=n_configs, loc='lower center',
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'improvement_chart.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_loss_curve(runs):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    agg = aggregate_history(runs, 'loss')
    for tag in DISPLAY_ORDER:
        if tag not in agg:
            continue
        ep, mean, std = agg[tag]
        display, color, marker = CONFIGS[tag]
        ax.plot(ep, mean, label=display, color=color, linewidth=1.5,
                marker=marker, markersize=3)
        if (std > 0).any():
            ax.fill_between(ep, mean - std, mean + std, color=color,
                            alpha=0.18, linewidth=0)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Training loss', fontsize=10)
    ax.set_title('Training Loss (mean ± 1σ across seeds)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'loss_curve.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_seed_variance(summary):
    plt = _setup_mpl()
    configs_present = [t for t in DISPLAY_ORDER if t in summary]
    if not configs_present:
        return

    fig, ax = plt.subplots(figsize=(8, 4.4))
    metric = 'NDCG@10'
    for i, tag in enumerate(configs_present):
        e = summary[tag]
        per_seed = e.get('test_per_seed', {})
        if not per_seed:
            continue
        display, color, marker = CONFIGS[tag]
        ys = [v[metric] for v in per_seed.values()]
        xs = np.full(len(ys), i) + np.random.uniform(-0.07, 0.07, len(ys))
        ax.scatter(xs, ys, color=color, s=55, marker=marker,
                   edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
        # Mean bar
        mu = e['test_mean'][metric]
        ax.hlines(mu, i - 0.18, i + 0.18, color=color, linewidth=2.5, zorder=2)

    ax.set_xticks(np.arange(len(configs_present)))
    ax.set_xticklabels([CONFIGS[t][0] for t in configs_present],
                       fontsize=9, rotation=15, ha='right')
    ax.set_ylabel(f'Test {metric}', fontsize=10)
    ax.set_title(f'Seed Variance per Config — {metric}', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'seed_variance.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ------------------------------------------------------------------ #
#  Summary persistence                                                 #
# ------------------------------------------------------------------ #
def save_summary(summary):
    path = os.path.join(RESULTS_DIR, 'comparison_summary.json')
    out = {CONFIGS[tag][0]: summary[tag] for tag in DISPLAY_ORDER if tag in summary}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSummary saved: {path}')


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    runs = discover_runs()
    n_total = sum(len(s) for s in runs.values())
    if n_total == 0:
        print(f'No matching result files found in {RESULTS_DIR}.')
        print('Expected pattern: {tag}_s{seed}.json with tag in '
              f'{list(CONFIGS.keys())}')
        return

    print(f'Discovered {n_total} run(s):')
    for tag in DISPLAY_ORDER:
        seeds = sorted(runs[tag].keys())
        if seeds:
            print(f'  {CONFIGS[tag][0]:<18} seeds={seeds}')

    summary = aggregate_test(runs)
    summary = add_improvements(summary)

    print_table(summary)
    print_improvements(summary)

    print('\nGenerating figures...')
    fig_convergence(runs)
    fig_results_bar(summary)
    fig_improvement(summary)
    fig_loss_curve(runs)
    fig_seed_variance(summary)

    save_summary(summary)


if __name__ == '__main__':
    main()
