"""
select-ts — filter tilt series by quality criteria from an analyse output.

Reads alignment_data.json from a previous analyse run, applies quality
filters, and writes a ts-select.csv file.

CSV columns:
  ts_name, selected, n_frames, n_tilts, rating, rot_deg, thickness_px,
  thickness_angst, thickness_nm, ref_defocus_um, alpha_deg, exclude_reason

  n_frames  total aligned frames in the .aln (AreTomo3 dark frames excluded)
  n_tilts   frames with overlap_pct >= --overlap-thres (usable after overlap
            filtering); equals n_frames when --overlap-thres is not given
  rating    star rating from ts_ratings.csv in the analysis dir (empty if
            not rated or file not present)
"""

import csv
import sys
from collections import Counter
from pathlib import Path
import argparse
import json

from aretomo3_preprocess.shared.project_json import load as load_project
from aretomo3_preprocess.shared.project_state import get_latest_analysis_dir


# ─────────────────────────────────────────────────────────────────────────────
# Ratings loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_ratings(analysis_dir: Path) -> dict:
    """Load ts_ratings.csv from analysis_dir; return {ts_name: int} or {}."""
    path = analysis_dir / 'ts_ratings.csv'
    if not path.exists():
        return {}
    ratings = {}
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ratings[row['ts_name']] = int(row['rating'])
            except (KeyError, ValueError):
                pass
    return ratings


# ─────────────────────────────────────────────────────────────────────────────
# Per-TS statistics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ts_stats(ts_name, ts_data, overlap_thres):
    """Compute per-TS summary statistics from an alignment_data.json entry."""
    frames       = ts_data.get('frames', [])
    n_frames     = len(frames)
    angpix       = ts_data.get('angpix')
    thickness_nm = ts_data.get('thickness_nm')

    if thickness_nm is not None and angpix is not None and angpix > 0:
        thickness_px    = round(thickness_nm * 10.0 / angpix, 1)
        thickness_angst = round(thickness_nm * 10.0, 1)
    else:
        thickness_px    = None
        thickness_angst = None

    if n_frames == 0:
        return {
            'ts_name':        ts_name,
            'n_frames':       0,
            'n_tilts':        0,
            'rot_deg':        None,
            'thickness_px':   thickness_px,
            'thickness_angst':thickness_angst,
            'thickness_nm':   thickness_nm,
            'ref_defocus_um': None,
            'alpha_deg':      ts_data.get('alpha_offset'),
        }

    rots    = [f['rot'] for f in frames if 'rot' in f]
    rot_deg = round(sum(rots) / len(rots), 3) if rots else None

    # n_tilts: frames not eliminated by overlap threshold
    if overlap_thres is not None:
        n_tilts = sum(
            1 for f in frames
            if f.get('overlap_pct') is None or f.get('overlap_pct') >= overlap_thres
        )
    else:
        n_tilts = n_frames

    # Defocus of first acquisition (acq_order == 1)
    acq1 = [f for f in frames if f.get('acq_order') == 1]
    if acq1:
        ref_frame = acq1[0]
    else:
        ref_frame = min(frames, key=lambda f: abs(f.get('tilt', 0)))
    ref_defocus = ref_frame.get('mean_defocus_um') if ref_frame else None

    return {
        'ts_name':        ts_name,
        'n_frames':       n_frames,
        'n_tilts':        n_tilts,
        'rot_deg':        rot_deg,
        'thickness_px':   thickness_px,
        'thickness_angst':thickness_angst,
        'thickness_nm':   thickness_nm,
        'ref_defocus_um': ref_defocus,
        'alpha_deg':      ts_data.get('alpha_offset'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filter evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(stats, args):
    """Return list of exclusion reasons (empty list = TS is selected)."""
    reasons = []

    if args.select_by_rating is not None:
        lo, hi = args.select_by_rating
        rating = stats.get('rating')
        if rating is None:
            reasons.append('unrated')
        elif rating < lo:
            reasons.append(f'rating<{lo:.0f}')
        elif rating > hi:
            reasons.append(f'rating>{hi:.0f}')

    if args.select_by_tilts is not None:
        lo, hi = args.select_by_tilts
        if stats['n_tilts'] < lo:
            reasons.append(f'tilts<{lo:.0f}')
        if stats['n_tilts'] > hi:
            reasons.append(f'tilts>{hi:.0f}')

    if args.select_by_tilt_axis is not None and stats['rot_deg'] is not None:
        lo, hi = args.select_by_tilt_axis
        if stats['rot_deg'] < lo:
            reasons.append(f'rot<{lo}')
        if stats['rot_deg'] > hi:
            reasons.append(f'rot>{hi}')

    if args.select_by_thickness_px is not None and stats['thickness_px'] is not None:
        lo, hi = args.select_by_thickness_px
        if stats['thickness_px'] < lo:
            reasons.append(f'thickness<{lo:.0f}px')
        if stats['thickness_px'] > hi:
            reasons.append(f'thickness>{hi:.0f}px')

    if args.select_by_thickness_angst is not None and stats['thickness_angst'] is not None:
        lo, hi = args.select_by_thickness_angst
        if stats['thickness_angst'] < lo:
            reasons.append(f'thickness<{lo:.0f}Å')
        if stats['thickness_angst'] > hi:
            reasons.append(f'thickness>{hi:.0f}Å')

    if args.select_by_defocus is not None and stats['ref_defocus_um'] is not None:
        lo, hi = args.select_by_defocus
        if stats['ref_defocus_um'] < lo:
            reasons.append(f'defocus<{lo}μm')
        if stats['ref_defocus_um'] > hi:
            reasons.append(f'defocus>{hi}μm')

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'select-ts',
        help='Filter tilt series by quality criteria; writes ts-select.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--analysis', '-A', default=None,
                   help='analyse output directory containing alignment_data.json. '
                        'Auto-read from project.json if omitted.')
    p.add_argument('--output', '-o', default='ts-select.csv',
                   help='Output CSV file path (default: ts-select.csv)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be selected/excluded without writing CSV')

    p.add_argument('--overlap-thres', type=float, default=None, metavar='PCT',
                   help='Minimum overlap_pct for a frame to count as a usable '
                        'tilt (sets n_tilts column and is used by '
                        '--select-by-tilts).  Warn if different from the '
                        'filter_overlap value used in the last run-aretomo3 run.')

    sel = p.add_argument_group('selection filters (all optional; provide MIN MAX)')
    sel.add_argument('--select-by-rating', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with star rating in [MIN, MAX] '
                          '(reads ts_ratings.csv from the analysis directory; '
                          'TS without a rating are excluded)')
    sel.add_argument('--select-by-tilts', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with n_tilts (usable frames after overlap '
                          'filtering) in [MIN, MAX]')
    sel.add_argument('--select-by-tilt-axis', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with mean tilt-axis rotation in [MIN, MAX] degrees')
    sel.add_argument('--select-by-thickness-px', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with estimated sample thickness in [MIN, MAX] pixels')
    sel.add_argument('--select-by-thickness-angst', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with estimated sample thickness in [MIN, MAX] Å')
    sel.add_argument('--select-by-defocus', type=float, nargs=2,
                     metavar=('MIN', 'MAX'),
                     help='Keep TS with first-acquisition defocus in [MIN, MAX] μm')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── Resolve analysis dir ─────────────────────────────────────────────────
    analysis_dir = args.analysis
    if analysis_dir is None:
        analysis_dir = get_latest_analysis_dir()
        if analysis_dir is not None:
            print(f'Note: --analysis not given; using project.json → {analysis_dir}')
    if analysis_dir is None:
        print('ERROR: --analysis not given and no analyse output found in project.json.')
        print('       Run analyse first, or supply --analysis explicitly.')
        sys.exit(1)

    analysis_dir = Path(analysis_dir)
    json_path    = analysis_dir / 'alignment_data.json'
    if not json_path.exists():
        print(f'ERROR: alignment_data.json not found in {analysis_dir}')
        sys.exit(1)

    with open(json_path) as f:
        alignment_data = json.load(f)

    ts_names = sorted(alignment_data.keys())
    print(f'Loaded {len(ts_names)} tilt series from {analysis_dir}/alignment_data.json')

    # ── Load ratings ─────────────────────────────────────────────────────────
    ratings = _load_ratings(analysis_dir)
    if ratings:
        print(f'Loaded ratings for {len(ratings)} tilt series from '
              f'{analysis_dir}/ts_ratings.csv')
    elif args.select_by_rating is not None:
        print(f'WARNING: --select-by-rating given but {analysis_dir}/ts_ratings.csv '
              f'not found — all TS will be excluded as unrated.')

    # ── Overlap threshold consistency check ──────────────────────────────────
    if args.overlap_thres is not None:
        proj         = load_project()
        saved_thres  = (proj.get('run_aretomo3', {})
                            .get('args', {})
                            .get('filter_overlap'))
        if saved_thres is not None and saved_thres != args.overlap_thres:
            print(f'WARNING: --overlap-thres {args.overlap_thres} differs from '
                  f'filter_overlap={saved_thres} used in the last run-aretomo3 run.')
            print(f'         n_tilts counts may not match the frames actually '
                  f'used in reconstruction.')
        print(f'Overlap threshold: {args.overlap_thres}%  '
              f'(frames below this are excluded from n_tilts)')
    print()

    # ── Apply filters ────────────────────────────────────────────────────────
    rows           = []
    selected_names = []
    n_excluded     = 0
    reason_counts  = Counter()

    for ts_name in ts_names:
        stats          = _compute_ts_stats(ts_name, alignment_data[ts_name],
                                           args.overlap_thres)
        stats['rating'] = ratings.get(ts_name)   # None if unrated
        reasons        = _apply_filters(stats, args)
        selected = len(reasons) == 0

        if selected:
            selected_names.append(ts_name)
        else:
            n_excluded += 1
            for r in reasons:
                reason_counts[r] += 1

        rows.append({
            'ts_name':        ts_name,
            'selected':       1 if selected else 0,
            'n_frames':       stats['n_frames'],
            'n_tilts':        stats['n_tilts'],
            'rating':         stats['rating'] if stats['rating'] is not None else '',
            'rot_deg':        _fmt(stats['rot_deg']),
            'thickness_px':   _fmt(stats['thickness_px']),
            'thickness_angst':_fmt(stats['thickness_angst']),
            'thickness_nm':   _fmt(stats['thickness_nm']),
            'ref_defocus_um': _fmt(stats['ref_defocus_um']),
            'alpha_deg':      _fmt(stats['alpha_deg']),
            'exclude_reason': '; '.join(reasons),
        })

    # ── Dry-run: print per-TS detail and stop ────────────────────────────────
    if args.dry_run:
        tag = '[DRY RUN] '
        w   = max(len(ts) for ts in ts_names)
        for row in rows:
            status = 'SELECT' if row['selected'] else f'EXCLUDE ({row["exclude_reason"]})'
            tilts  = f'{row["n_tilts"]}/{row["n_frames"]} tilts'
            stars  = (f'  {row["rating"]}★' if row['rating'] != '' else '  -★')
            thick  = (f'  {row["thickness_px"]}px / {row["thickness_angst"]}Å'
                      if row['thickness_px'] != '' else '')
            defoc  = (f'  defocus={row["ref_defocus_um"]}μm'
                      if row['ref_defocus_um'] != '' else '')
            rot    = (f'  rot={row["rot_deg"]}°' if row['rot_deg'] != '' else '')
            print(f'{tag}{row["ts_name"]:{w}}  {status:<35}  {tilts}{stars}{thick}{defoc}{rot}')
        print()
        print(f'{tag}Would select {len(selected_names)} / {len(ts_names)}  '
              f'({n_excluded} excluded)')
        if n_excluded and reason_counts:
            print()
            print(f'{tag}Exclusion reasons:')
            for reason, count in sorted(reason_counts.items()):
                print(f'{tag}  {reason}: {count}')
        return

    # ── Write CSV ────────────────────────────────────────────────────────────
    out_path   = Path(args.output)
    fieldnames = [
        'ts_name', 'selected', 'n_frames', 'n_tilts', 'rating',
        'rot_deg', 'thickness_px', 'thickness_angst', 'thickness_nm',
        'ref_defocus_um', 'alpha_deg', 'exclude_reason',
    ]
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Selected : {len(selected_names)} / {len(ts_names)}  '
          f'({n_excluded} excluded)')
    print(f'Written  : {out_path}')

    if n_excluded and reason_counts:
        print()
        print('Exclusion reasons:')
        for reason, count in sorted(reason_counts.items()):
            print(f'  {reason}: {count}')


def _fmt(v):
    if v is None:
        return ''
    return round(v, 3) if isinstance(v, float) else v
