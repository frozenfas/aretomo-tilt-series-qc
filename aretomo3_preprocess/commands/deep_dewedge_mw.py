"""
ddw-train-mw  — Train separate DeepDeWedge models per missing-wedge-angle bin.
ddw-refine-mw — Apply per-bin models to denoise tomograms.

Extends ddw-train / ddw-refine by computing the actual missing wedge angle for
each tilt series from its .aln file, grouping tilt series into bins, and training
one DDW model per bin.

Motivation
----------
DDW uses mw_angle as a core parameter for generating the missing-wedge mask
during training.  If tilt series have meaningfully different tilt ranges (due
to dark-frame exclusion, overlap filtering at high tilts, or aborted acquisitions),
a single mw_angle is a compromise.  Per-bin training allows each model to learn
the correct missing-wedge geometry for its group.

Workflow
--------
  # Step 1: explore the distribution (dry-run, no bins needed)
  aretomo3-preprocess ddw-train-mw \\
      --input run002-cmd2-sart-thr80/ \\
      --output run002-ddw-mw/ \\
      --dry-run

  # Step 2: train with chosen bin edges (from the histogram output)
  aretomo3-preprocess ddw-train-mw \\
      --input run002-cmd2-sart-thr80/ \\
      --output run002-ddw-mw/ \\
      --mw-edges 44 48 \\
      --gpu 0 1

  # Step 3: refine all TS using per-bin models
  aretomo3-preprocess ddw-refine-mw \\
      --input run002-cmd2-sart-thr80/ \\
      --project-dir run002-ddw-mw/ \\
      --gpu 0
"""

import sys
import json
import shutil
import datetime
from collections import Counter, defaultdict
from pathlib import Path
import argparse

import yaml

from aretomo3_preprocess.shared.project_json import (
    update_section, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.commands.cryocare import _find_evn_odd_pairs
from aretomo3_preprocess.commands.deep_dewedge import (
    _find_best_checkpoint, _write_yaml, _run_ddw,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-TS missing wedge calculation
# ─────────────────────────────────────────────────────────────────────────────

def _mw_from_aln(aln_path: Path):
    """
    Compute DDW mw_angle from the tilt angles in a single .aln file.

    DDW convention (from missing_wedge.py):
        alpha = mw_angle / 2
        mask boundary from Z-axis = 90 - alpha = 90 - mw_angle/2

    For the mask boundary to equal the actual missing wedge edge (= 90 - theta_max):
        mw_angle = 2 * theta_max

    For asymmetric tilt ranges (different positive and negative extents), the
    conservative choice covers the side with the larger missing wedge:
        mw_angle = 2 * min(|max_pos_tilt|, |max_neg_tilt|)

    Returns None on failure.
    """
    max_pos = 0.0   # furthest positive tilt
    max_neg = 0.0   # furthest negative tilt (stored positive)
    found   = False
    try:
        with open(aln_path) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith('#') or not stripped:
                    continue
                parts = stripped.split()
                if len(parts) == 10:
                    try:
                        tilt    = float(parts[9])
                        max_pos = max(max_pos,  tilt)
                        max_neg = max(max_neg, -tilt)
                        found   = True
                    except ValueError:
                        pass
    except OSError:
        return None
    if not found:
        return None
    # Use the smaller of the two max tilts (conservative — covers the side
    # with the larger missing wedge).
    effective_max_tilt = min(max_pos, max_neg)
    return max(0, int(round(2.0 * effective_max_tilt)))


def _mw_per_ts(in_dir: Path) -> dict:
    """
    Compute mw_angle for every ts-*.aln in in_dir.
    Returns {ts_name: mw_angle_int}.  TS without .aln are omitted.
    """
    result = {}
    for aln_path in sorted(in_dir.glob('ts-*.aln')):
        mw = _mw_from_aln(aln_path)
        if mw is not None:
            result[aln_path.stem] = mw   # stem = ts-XXX
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Histogram and bin helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_mw_histogram(mw_angles: dict, bar_width: int = 40):
    values = list(mw_angles.values())
    if not values:
        print('  (no data)')
        return

    counts  = Counter(values)
    lo, hi  = min(counts), max(counts)
    max_cnt = max(counts.values())
    n       = len(values)
    mean    = sum(values) / n
    median  = sorted(values)[n // 2]
    std     = (sum((v - mean) ** 2 for v in values) / n) ** 0.5

    print(f'Missing wedge angle distribution  ({n} TS):')
    print()
    for angle in range(lo, hi + 1):
        cnt     = counts.get(angle, 0)
        bar_len = int(round(cnt / max_cnt * bar_width)) if max_cnt else 0
        pct     = 100.0 * cnt / n
        print(f'  {angle:3d}°  |{"█" * bar_len:<{bar_width}}|  {cnt:4d}  ({pct:5.1f}%)')
    print()
    print(f'  min={lo}°  max={hi}°  mean={mean:.1f}°  '
          f'median={median}°  std={std:.1f}°')


def _save_histogram_png(mw_angles: dict, out_path: Path):
    """Save a matplotlib histogram. Silent no-op if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    values  = list(mw_angles.values())
    lo, hi  = min(values), max(values)
    bin_edges = [b - 0.5 for b in range(lo, hi + 2)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bin_edges, edgecolor='black', color='steelblue', alpha=0.8)
    ax.set_xlabel('Missing wedge angle (°)')
    ax.set_ylabel('Number of tilt series')
    ax.set_title(f'Missing wedge distribution  ({len(values)} TS)')
    ax.set_xticks(range(lo, hi + 1))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Histogram PNG   : {out_path}')


def _suggest_splits(mw_angles: dict):
    """Print suggested --mw-edges values based on quartile splits."""
    values = sorted(mw_angles.values())
    n = len(values)
    if n < 4:
        return
    print('Suggested splits (quartile-based):')
    for n_bins in [2, 3, 4]:
        edges = []
        for i in range(1, n_bins):
            idx  = int(round(i * n / n_bins))
            idx  = max(1, min(idx, n - 1))
            edge = (values[idx - 1] + values[idx]) // 2 + 1
            edges.append(edge)
        edges = sorted(set(edges))
        bins  = _assign_bins(mw_angles, edges)
        sizes = [len(v) for v in bins.values() if v]
        edge_str = ' '.join(str(e) for e in edges) if edges else '(none)'
        print(f'  {n_bins} bins  --mw-edges {edge_str:<15}  '
              f'→  {" / ".join(str(s) + " TS" for s in sizes)}')


def _assign_bins(mw_angles: dict, edges: list) -> dict:
    """
    Assign TS to bins defined by sorted edge list.

    Bin i contains TS where edges[i-1] <= mw_angle < edges[i].
    First bin:  mw_angle < edges[0]
    Last bin:   mw_angle >= edges[-1]

    Returns OrderedDict {bin_label: [ts_name, ...]} in ascending order.
    Empty bins are included (so label set is always the same for given edges).
    """
    edges = sorted(int(e) for e in edges)
    n     = len(edges)

    def _label(i):
        if n == 0:
            return 'bin0_all'
        if i == 0:
            return f'bin0_lt{edges[0]}'
        if i == n:
            return f'bin{i}_ge{edges[-1]}'
        return f'bin{i}_{edges[i-1]}to{edges[i]}'

    bins = {_label(i): [] for i in range(n + 1)}
    for ts_name, mw in sorted(mw_angles.items()):
        idx = sum(1 for e in edges if mw >= e)
        bins[_label(idx)].append(ts_name)
    return bins


# ─────────────────────────────────────────────────────────────────────────────
# ddw-train-mw
# ─────────────────────────────────────────────────────────────────────────────

def _add_train_mw_parser(subparsers):
    p = subparsers.add_parser(
        'ddw-train-mw',
        help='Train one DDW model per missing-wedge-angle bin [dev]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory with ts-*_EVN_Vol.mrc / _ODD_Vol.mrc '
                          'and ts-*.aln files')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume suffix; empty=auto-detect')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only selected TS are included')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', required=True,
                     help='Base output directory; one subdir per bin is created here')

    bns = p.add_argument_group('binning (mutually exclusive)')
    mwg = bns.add_mutually_exclusive_group()
    mwg.add_argument('--mw-bins', type=int, default=None, metavar='N',
                     help='Split into N equal-width bins over the observed range. '
                          'Omit in --dry-run to see the histogram first.')
    mwg.add_argument('--mw-edges', type=float, nargs='+', metavar='DEG',
                     help='Explicit bin edges (degrees, ascending). '
                          'N edges → N+1 bins.')

    ddw = p.add_argument_group('DeepDeWedge parameters')
    ddw.add_argument('--subtomo-size', type=int, default=96,
                     help='Subtomogram box size in voxels')
    ddw.add_argument('--num-epochs',   type=int,   default=1000)
    ddw.add_argument('--batch-size',   type=int,   default=5)
    ddw.add_argument('--lr',           type=float, default=0.0004)
    ddw.add_argument('--unet-chans',   type=int,   default=64)
    ddw.add_argument('--unet-depth',   type=int,   default=3)
    ddw.add_argument('--val-fraction', type=float, default=0.1)
    ddw.add_argument('--standardize',  action='store_true', default=False,
                     help='Standardize full tomos before subtomogram extraction')
    ddw.add_argument('--num-workers',  type=int,   default=8)

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, nargs='+', default=[0], metavar='ID')
    ctl.add_argument('--seed', type=int, default=42)
    ctl.add_argument('--ddw', dest='ddw_bin',
                     default='/opt/miniconda3/envs/ddw_env/bin/ddw',
                     help='Path to ddw executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Show histogram / planned splits without running DDW; '
                          'if no bins given, stops after histogram')

    p.set_defaults(func=_run_train_mw)
    return p


def _run_train_mw(args):
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    sep     = '─' * 70
    prefix  = '[DRY RUN] ' if args.dry_run else ''

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # ── Find EVN/ODD pairs ─────────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no EVN/ODD volume pairs found in {in_dir}/')
        print('       Run-aretomo3 cmd=2 with --split-sum 1 first.')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Apply TS selection filter ──────────────────────────────────────────
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        orig_n = len(pairs)
        pairs  = [p for p in pairs if p['ts_name'] in selected_ts]
        n_excl = orig_n - len(pairs)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(pairs)} remaining')
        if not pairs:
            print('ERROR: no volumes remain after TS selection filter')
            sys.exit(1)

    pair_names = {p['ts_name'] for p in pairs}

    # ── Compute per-TS mw_angle ────────────────────────────────────────────
    print(f'Reading .aln files from {in_dir}/')
    all_mw    = _mw_per_ts(in_dir)
    mw_angles = {ts: all_mw[ts] for ts in pair_names if ts in all_mw}
    missing   = [ts for ts in sorted(pair_names) if ts not in all_mw]
    if missing:
        print(f'WARNING: no .aln for {len(missing)} TS (will be excluded):')
        for ts in missing[:10]:
            print(f'  {ts}')
        if len(missing) > 10:
            print(f'  ... (+{len(missing) - 10} more)')
    print(f'mw_angle computed for {len(mw_angles)} / {len(pairs)} TS')
    print(sep)

    # ── Histogram ─────────────────────────────────────────────────────────
    _print_mw_histogram(mw_angles)
    print()
    _suggest_splits(mw_angles)
    print()

    # Save histogram PNG (writes to disk even in dry-run if dir already exists,
    # or always if not dry-run)
    hist_path = out_dir / 'mw_histogram.png'
    if args.dry_run:
        if out_dir.exists():
            _save_histogram_png(mw_angles, hist_path)
        else:
            print(f'{prefix}Histogram PNG would be written to {hist_path}')
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_histogram_png(mw_angles, hist_path)

    # ── If no bins given: histogram-only mode ─────────────────────────────
    if args.mw_edges is None and args.mw_bins is None:
        if args.dry_run:
            print('(Dry-run: no --mw-bins / --mw-edges given — histogram only)')
            return
        print('ERROR: specify --mw-bins N or --mw-edges A B C ... to train.')
        sys.exit(1)

    # ── Determine bin edges ────────────────────────────────────────────────
    if args.mw_edges is not None:
        edges = sorted(int(e) for e in args.mw_edges)
    else:
        lo, hi = min(mw_angles.values()), max(mw_angles.values())
        if args.mw_bins == 1 or lo == hi:
            edges = []
        else:
            span  = hi - lo
            step  = span / args.mw_bins
            edges = sorted(set(int(round(lo + step * i))
                               for i in range(1, args.mw_bins)))

    # ── Assign TS to bins ──────────────────────────────────────────────────
    bins = _assign_bins(mw_angles, edges)
    # Drop empty bins (e.g. edge outside observed range)
    bins = {lbl: ts_list for lbl, ts_list in bins.items() if ts_list}

    print(sep)
    print(f'Bin assignments  ({len(bins)} bins, edges: {edges if edges else "none"}):')
    for label, ts_list in bins.items():
        vals    = [mw_angles[ts] for ts in ts_list]
        mw_med  = sorted(vals)[len(vals) // 2]
        print(f'  {label:22s}  {len(ts_list):3d} TS  '
              f'mw range {min(vals)}–{max(vals)}°  median {mw_med}°')
    print(sep)

    # Validate subtomo-size
    div = 2 ** args.unet_depth
    if args.subtomo_size % div != 0:
        print(f'ERROR: --subtomo-size {args.subtomo_size} not divisible by '
              f'2^{args.unet_depth} = {div}')
        sys.exit(1)

    # ── Dry-run: show plan and exit ────────────────────────────────────────
    if args.dry_run:
        print(f'{prefix}Would write: {out_dir}/ts_mw_assignments.json')
        for label, ts_list in bins.items():
            mw_med = sorted([mw_angles[ts] for ts in ts_list])[len(ts_list) // 2]
            print(f'{prefix}Would train: {out_dir / label}/  '
                  f'(mw_angle={mw_med}°, {len(ts_list)} TS)')
        return

    # ── Write ts_mw_assignments.json ──────────────────────────────────────
    ts_to_bin = {ts: lbl for lbl, ts_list in bins.items() for ts in ts_list}
    assignments = {
        ts: {'mw_angle': mw_angles[ts], 'bin': ts_to_bin.get(ts)}
        for ts in sorted(mw_angles)
    }
    assign_data = {
        'bin_edges':   edges,
        'bins':        {lbl: ts_list for lbl, ts_list in bins.items()},
        'assignments': assignments,
    }
    assign_path = out_dir / 'ts_mw_assignments.json'
    with open(assign_path, 'w') as fh:
        json.dump(assign_data, fh, indent=2)
    print(f'Written: {assign_path}')

    gpu_val  = args.gpu[0] if len(args.gpu) == 1 else args.gpu
    pair_lut = {p['ts_name']: p for p in pairs}

    # ── Train one model per bin ────────────────────────────────────────────
    for label, ts_list in bins.items():
        bin_dir    = out_dir / label
        bin_pairs  = [pair_lut[ts] for ts in ts_list if ts in pair_lut]
        mw_vals    = [mw_angles[ts] for ts in ts_list]
        mw_med     = sorted(mw_vals)[len(mw_vals) // 2]

        print(f'\n{"═" * 70}')
        print(f'Bin: {label}  ({len(bin_pairs)} TS,  median mw_angle = {mw_med}°)')
        print(f'{"═" * 70}')

        bin_dir.mkdir(parents=True, exist_ok=True)

        config = {
            'shared': {
                'project_dir':  str(bin_dir.resolve()),
                'tomo0_files':  [str(p['evn'].resolve()) for p in bin_pairs],
                'tomo1_files':  [str(p['odd'].resolve()) for p in bin_pairs],
                'subtomo_size': args.subtomo_size,
                'mw_angle':     mw_med,
                'num_workers':  args.num_workers,
                'gpu':          gpu_val,
                'seed':         args.seed,
            },
            'prepare_data': {
                'val_fraction':           args.val_fraction,
                'standardize_full_tomos': args.standardize,
            },
            'fit_model': {
                'unet_params_dict': {
                    'chans':                 args.unet_chans,
                    'num_downsample_layers': args.unet_depth,
                    'drop_prob':             0.0,
                },
                'adam_params_dict': {'lr': args.lr},
                'num_epochs':  args.num_epochs,
                'batch_size':  args.batch_size,
                'update_subtomo_missing_wedges_every_n_epochs': 10,
                'check_val_every_n_epochs':                     10,
                'save_n_models_with_lowest_val_loss':           5,
                'save_n_models_with_lowest_fitting_loss':       5,
                'save_model_every_n_epochs':                    50,
                'logger': 'csv',
            },
        }
        config_path = bin_dir / 'ddw_config.yaml'
        _write_yaml(config_path, config)

        if shutil.which(args.ddw_bin) is None and not Path(args.ddw_bin).is_file():
            print(f'ERROR: ddw binary not found: {args.ddw_bin!r}')
            sys.exit(1)

        print('Step 1: prepare-data')
        _run_ddw(args.ddw_bin, 'prepare-data', config_path)
        print('Step 2: fit-model')
        _run_ddw(args.ddw_bin, 'fit-model', config_path)

    print(f'\n{"═" * 70}')
    print(f'All bins trained.')
    print(f'  Project dir : {out_dir}')
    print(f'\n  Next step:')
    print(f'    aretomo3-preprocess ddw-refine-mw \\')
    print(f'        --input {in_dir} --project-dir {out_dir} --gpu 0')

    update_section(
        section='ddw_train_mw',
        values={
            'command':    ' '.join(sys.argv),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'output_dir': str(out_dir.resolve()),
            'bin_edges':  edges,
            'bins':       {lbl: len(ts_list) for lbl, ts_list in bins.items()},
        },
        backup_dir=out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ddw-refine-mw
# ─────────────────────────────────────────────────────────────────────────────

def _add_refine_mw_parser(subparsers):
    p = subparsers.add_parser(
        'ddw-refine-mw',
        help='Refine tomograms using per-mw-bin DDW models [dev]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory with ts-*_EVN_Vol.mrc / _ODD_Vol.mrc')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume suffix; empty=auto-detect')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only selected TS are refined')
    inp.add_argument('--project-dir', required=True,
                     help='ddw-train-mw output directory '
                          '(contains ts_mw_assignments.json)')

    ddw = p.add_argument_group('DeepDeWedge parameters')
    ddw.add_argument('--batch-size',  type=int, default=10)
    ddw.add_argument('--num-workers', type=int, default=4)

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, default=0, metavar='ID')
    ctl.add_argument('--ddw', dest='ddw_bin',
                     default='/opt/miniconda3/envs/ddw_env/bin/ddw',
                     help='Path to ddw executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write configs without running ddw')

    p.set_defaults(func=_run_refine_mw)
    return p


def _run_refine_mw(args):
    in_dir      = Path(args.input)
    project_dir = Path(args.project_dir)
    sep         = '─' * 70
    prefix      = '[DRY RUN] ' if args.dry_run else ''

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)
    if not project_dir.is_dir():
        print(f'ERROR: --project-dir {project_dir} not found')
        sys.exit(1)

    # ── Load mw assignments from training run ─────────────────────────────
    assign_path = project_dir / 'ts_mw_assignments.json'
    if not assign_path.exists():
        print(f'ERROR: {assign_path} not found. Run ddw-train-mw first.')
        sys.exit(1)
    with open(assign_path) as fh:
        assign_data = json.load(fh)

    edges       = assign_data.get('bin_edges', [])
    saved_assn  = assign_data.get('assignments', {})   # ts_name → {mw_angle, bin}

    # ── Find EVN/ODD pairs ─────────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no EVN/ODD pairs found in {in_dir}/')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Apply TS selection filter ──────────────────────────────────────────
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        orig_n = len(pairs)
        pairs  = [p for p in pairs if p['ts_name'] in selected_ts]
        n_excl = orig_n - len(pairs)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(pairs)} remaining')
        if not pairs:
            print('ERROR: no volumes remain after TS selection filter')
            sys.exit(1)

    # ── Determine mw_angle for each pair ──────────────────────────────────
    # Prefer .aln-derived value; fall back to stored assignment.
    print(f'Reading .aln files from {in_dir}/')
    aln_mw = _mw_per_ts(in_dir)

    mw_for_pairs = {}
    for p in pairs:
        ts = p['ts_name']
        if ts in aln_mw:
            mw_for_pairs[ts] = aln_mw[ts]
        elif ts in saved_assn and saved_assn[ts].get('mw_angle') is not None:
            mw_for_pairs[ts] = saved_assn[ts]['mw_angle']

    no_mw = [p['ts_name'] for p in pairs if p['ts_name'] not in mw_for_pairs]
    if no_mw:
        print(f'WARNING: no mw_angle for {len(no_mw)} TS — they will be skipped:')
        for ts in no_mw[:10]:
            print(f'  {ts}')

    # ── Assign pairs to bins using stored edges ────────────────────────────
    bins = _assign_bins(mw_for_pairs, edges)
    bins = {lbl: ts_list for lbl, ts_list in bins.items() if ts_list}

    print(sep)
    print(f'Bin assignments  ({len(bins)} bins):')
    for label, ts_list in bins.items():
        vals = [mw_for_pairs[ts] for ts in ts_list]
        print(f'  {label:22s}  {len(ts_list):3d} TS  '
              f'mw {min(vals)}–{max(vals)}°')
    print(sep)

    pair_lut = {p['ts_name']: p for p in pairs}

    # ── Refine each bin ────────────────────────────────────────────────────
    for label, ts_list in bins.items():
        bin_dir    = project_dir / label
        bin_pairs  = [pair_lut[ts] for ts in ts_list if ts in pair_lut]

        if not bin_dir.is_dir():
            print(f'WARNING: {bin_dir} not found — '
                  f'skipping {label} ({len(bin_pairs)} TS)')
            continue

        # Load train config for mw_angle, subtomo_size, standardize
        train_cfg_path = bin_dir / 'ddw_config.yaml'
        train_cfg = {}
        if train_cfg_path.exists():
            with open(train_cfg_path) as fh:
                train_cfg = yaml.safe_load(fh) or {}

        mw_angle     = train_cfg.get('shared', {}).get('mw_angle', 48)
        subtomo_size = train_cfg.get('shared', {}).get('subtomo_size', 96)
        standardize  = train_cfg.get('prepare_data', {}).get('standardize_full_tomos', False)

        # Find best checkpoint
        logdir = bin_dir / 'logs'
        if args.dry_run:
            ckpt_path = bin_dir / 'logs' / '<best_val_loss>.ckpt'
        else:
            ckpt_path = _find_best_checkpoint(logdir) if logdir.exists() else None
            if ckpt_path is None:
                print(f'WARNING: no checkpoint in {logdir} — '
                      f'skipping {label}  (run ddw-train-mw first?)')
                continue

        out_dir     = bin_dir / 'refined'
        config_path = bin_dir / 'ddw_refine_config.yaml'

        config = {
            'shared': {
                'project_dir':  str(bin_dir.resolve()),
                'tomo0_files':  [str(p['evn'].resolve()) for p in bin_pairs],
                'tomo1_files':  [str(p['odd'].resolve()) for p in bin_pairs],
                'subtomo_size': subtomo_size,
                'mw_angle':     mw_angle,
                'num_workers':  args.num_workers,
                'gpu':          args.gpu,
            },
            'refine_tomogram': {
                'model_checkpoint_file':  str(ckpt_path),
                'output_dir':             str(out_dir.resolve()),
                'batch_size':             args.batch_size,
                'subtomo_overlap':        subtomo_size // 3,
                'recompute_normalization': True,
                'standardize_full_tomos': standardize,
            },
        }

        print(f'\n{"═" * 70}')
        print(f'Bin: {label}  ({len(bin_pairs)} TS,  mw_angle = {mw_angle}°)')
        print(f'{prefix}Config  : {config_path}')
        print(f'{prefix}Checkpoint : {Path(str(ckpt_path)).name}')
        print(f'{prefix}Output  : {out_dir}/')
        print(f'{"═" * 70}')

        if not args.dry_run:
            _write_yaml(config_path, config)
            if shutil.which(args.ddw_bin) is None and not Path(args.ddw_bin).is_file():
                print(f'ERROR: ddw binary not found: {args.ddw_bin!r}')
                sys.exit(1)

        print(f'{prefix}Step 3: refine-tomogram')
        _run_ddw(args.ddw_bin, 'refine-tomogram', config_path, dry_run=args.dry_run)

    if args.dry_run:
        return

    print(f'\n{"═" * 70}')
    print(f'All bins refined.')
    print(f'  Denoised volumes in <project-dir>/<bin>/refined/')


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    _add_train_mw_parser(subparsers)
    _add_refine_mw_parser(subparsers)
