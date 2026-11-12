"""Plot already-published per-seed validation curves without changing values.

Input: {seed: {method: [{"labels": positive-int, "nrmse": finite-float}, ...]}}.
No smoothing, interpolation, averaging, training, or live artifact lookup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_curves(path):
    curves = json.loads(Path(path).read_text())
    if not isinstance(curves, dict) or not curves:
        raise ValueError('nonempty seed/method curve mapping required')
    for seed, methods in curves.items():
        if not isinstance(seed, str) or not seed.isdigit() or not isinstance(methods, dict) or not methods:
            raise ValueError('each seed must map to named methods')
        for method, rows in methods.items():
            if not isinstance(method, str) or not method or not isinstance(rows, list) or not rows:
                raise ValueError('named method with nonempty stored rows required')
            previous = 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {'labels', 'nrmse'}:
                    raise ValueError('only published labels/nrmse aggregate rows are accepted')
                budget, error = row['labels'], row['nrmse']
                if type(budget) is not int or budget <= previous:
                    raise ValueError('stored budgets must be strictly increasing positive integers')
                if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                    raise ValueError('stored NRMSE must be finite and nonnegative')
                previous = budget
    return curves


def independent_output(path):
    if __package__:
        from .history_restore import protected_worktrees
    else:
        from history_restore import protected_worktrees
    requested = Path(path).absolute()
    if requested.exists() or requested.is_symlink():
        raise ValueError('plot output must be a new file')
    if any(parent.is_symlink() for parent in requested.parents):
        raise ValueError('plot output parents must not be symlinks')
    output = requested.resolve()
    roots, _ = protected_worktrees(ROOT)
    if any(output == root or root in output.parents for root in roots):
        raise PermissionError('plot output must be outside all source worktrees')
    if output.suffix.lower() not in {'.png', '.svg', '.pdf'}:
        raise ValueError('plot output must be PNG, SVG or PDF')
    return output


def plot(input_path, output_path):
    curves = load_curves(input_path)
    output = independent_output(output_path)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    seeds = sorted(curves, key=int)
    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 3 * len(seeds)), squeeze=False)
    for seed, axes_row in zip(seeds, axes):
        ax = axes_row[0]
        for method, rows in curves[seed].items():
            ax.plot([r['labels'] for r in rows], [r['nrmse'] for r in rows], marker='o', ms=2, label=method)
        ax.set(title='Seed ' + seed, xlabel='Labeled training rows', ylabel='Stored validation NRMSE')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('Saved validation curves — raw, per seed')
    fig.text(.5, .008, 'Budgets/model prefixes can differ by method; consult the source provenance. No smoothing or extrapolation.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {'output': str(output), 'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            'seeds': seeds, 'rows': sum(len(rows) for methods in curves.values() for rows in methods.values()),
            'transformation': 'none; stored points plotted in their original per-method order'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.input, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
