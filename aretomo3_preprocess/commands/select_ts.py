"""
select-ts — filter tilt series by quality criteria from an analyse output.

Reads alignment_data.json from a previous analyse run, applies quality
filters, and writes a ts_selection.csv file.  The selection is also saved
to aretomo3_project.json so downstream commands (run-aretomo3,
run-aretomo3-per-ts, cryocare) can pick it up automatically.

CSV columns:
  ts_name, selected, n_frames, n_neg_frames, n_pos_frames, alpha_deg,
  thickness_nm, rot_deg, ref_defocus_um, exclude_reason

The tilt-balance filters (--min-neg-frames, --min-pos-frames) are the most
important: AreTomo3 crashes during WBP reconstruction when there are fewer
than ~2 frames on either the positive or negative tilt side.
"""

import csv
import sys
import datetime
from collections import Counter
from pathlib import Path
import argparse
import json

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import get_latest_analysis_dir


# ─────────────────────────────────────────────────────────────────────────────
# Per-TS statistics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ts_stats(ts_name, ts_data, tilt_threshold):
    """Compute per-TS summary statistics from an alignment_data.json entry."""
    frames = ts_data.get('frames', [])
    n_frames = len(frames)

    if n_frames == 0:
        return {
            'ts_name':        ts_name,
            'n_frames':       0,
            'n_neg_frames':   0,
            'n_pos_frames':   0,
            'alpha_deg':      ts_data.get('alpha_offset'),
            'thickness_nm':   ts_data.get('thickness_nm'),
            'rot_deg':        None,
            'ref_defocus_um': None,
        }

    tilts = [f['tilt'] for f in frames]
    rots  = [f['rot']  for f in frames]

    n_neg = sum(1 for t in tilts if t < -tilt_threshold)
    n_pos = sum(1 for t in tilts if t >  tilt_threshold)
    rot_deg = round(sum(rots) / len(rots), 3) if rots else None

    # Reference-frame defocus: prefer first-acquired frame (acq_order=1),
    # then is_reference=True, then frame closest to 0°.
    ref_frame = None
    acq1 = [f for f in frames if f.get('acq_order') == 1]
    if acq1:
        ref_frame = acq1[0]
    else:
        ref_list = [f for f in frames if f.get('is_reference')]
        if ref_list:
            ref_frame = ref_list[0]
        else:
            ref_frame = min(frames, key=lambda f: abs(f['tilt']))

    ref_defocus = ref_frame.get('mean_defocus_um') if ref_frame else None

    return {
        'ts_name':        ts_name,
        'n_frames':       n_frames,
        'n_neg_frames':   n_neg,
        'n_pos_frames':   n_pos,
        'alpha_deg':      ts_data.get('alpha_offset'),
        'thickness_nm':   ts_data.get('thickness_nm'),
        'rot_deg':        rot_deg,
        'ref_defocus_um': ref_defocus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filter evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(stats, args):
    """Return list of exclusion reasons (empty list = TS is selected)."""
    reasons = []

    if args.min_frames is not None and stats['n_frames'] < args.min_frames:
        reasons.append(f'frames<{args.min_frames}')
    if args.max_frames is not None and stats['n_frames'] > args.max_frames:
        reasons.append(f'frames>{args.max_frames}')

    if stats['n_neg_frames'] < args.min_neg_frames:
        reasons.append(f'neg_frames<{args.min_neg_frames}')
    if stats['n_pos_frames'] < args.min_pos_frames:
        reasons.append(f'pos_frames<{args.min_pos_frames}')

    if args.min_alpha is not None and stats['alpha_deg'] is not None:
        if stats['alpha_deg'] < args.min_alpha:
            reasons.append(f'alpha<{args.min_alpha}')
    if args.max_alpha is not None and stats['alpha_deg'] is not None:
        if stats['alpha_deg'] > args.max_alpha:
            reasons.append(f'alpha>{args.max_alpha}')

    if args.min_thickness is not None and stats['thickness_nm'] is not None:
        if stats['thickness_nm'] < args.min_thickness:
            reasons.append(f'thickness<{args.min_thickness}nm')
    if args.max_thickness is not None and stats['thickness_nm'] is not None:
        if stats['thickness_nm'] > args.max_thickness:
            reasons.append(f'thickness>{args.max_thickness}nm')

    if args.min_defocus is not None and stats['ref_defocus_um'] is not None:
        if stats['ref_defocus_um'] < args.min_defocus:
            reasons.append(f'defocus<{args.min_defocus}um')
    if args.max_defocus is not None and stats['ref_defocus_um'] is not None:
        if stats['ref_defocus_um'] > args.max_defocus:
            reasons.append(f'defocus>{args.max_defocus}um')

    if args.min_rot is not None and stats['rot_deg'] is not None:
        if stats['rot_deg'] < args.min_rot:
            reasons.append(f'rot<{args.min_rot}')
    if args.max_rot is not None and stats['rot_deg'] is not None:
        if stats['rot_deg'] > args.max_rot:
            reasons.append(f'rot>{args.max_rot}')

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'select-ts',
        help='Filter tilt series by quality criteria; writes ts_selection.csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--analysis', '-A', default=None,
                   help='analyse output directory containing alignment_data.json. '
                        'Auto-read from project.json if omitted.')
    p.add_argument('--output', '-o', default='ts_selection.csv',
                   help='Output CSV file path')

    bal = p.add_argument_group('tilt balance (prevents AreTomo3 WBP crashes)')
    bal.add_argument('--tilt-threshold', type=float, default=2.0,
                     help='Frames with |tilt| <= this are not counted as '
                          'positive or negative (degrees)')
    bal.add_argument('--min-neg-frames', type=int, default=2,
                     help='Minimum frames with tilt < -threshold')
    bal.add_argument('--min-pos-frames', type=int, default=2,
                     help='Minimum frames with tilt > +threshold')
    bal.add_argument('--min-frames', type=int, default=None,
                     help='Minimum total aligned frames')
    bal.add_argument('--max-frames', type=int, default=None,
                     help='Maximum total aligned frames')

    qfil = p.add_argument_group('quality filters (optional)')
    qfil.add_argument('--min-alpha', type=float, default=None,
                      help='Minimum alpha offset (tilt axis offset) in degrees')
    qfil.add_argument('--max-alpha', type=float, default=None,
                      help='Maximum alpha offset in degrees')
    qfil.add_argument('--min-thickness', type=float, default=None,
                      help='Minimum estimated sample thickness in nm')
    qfil.add_argument('--max-thickness', type=float, default=None,
                      help='Maximum estimated sample thickness in nm')
    qfil.add_argument('--min-defocus', type=float, default=None,
                      help='Minimum reference-frame defocus in μm')
    qfil.add_argument('--max-defocus', type=float, default=None,
                      help='Maximum reference-frame defocus in μm')
    qfil.add_argument('--min-rot', type=float, default=None,
                      help='Minimum mean tilt-axis rotation (rot) in degrees')
    qfil.add_argument('--max-rot', type=float, default=None,
                      help='Maximum mean tilt-axis rotation (rot) in degrees')

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

    analysis_dir   = Path(analysis_dir)
    json_path      = analysis_dir / 'alignment_data.json'
    if not json_path.exists():
        print(f'ERROR: alignment_data.json not found in {analysis_dir}')
        sys.exit(1)

    with open(json_path) as f:
        alignment_data = json.load(f)

    ts_names = sorted(alignment_data.keys())
    print(f'Loaded {len(ts_names)} tilt series from {analysis_dir}/alignment_data.json')
    print()

    # ── Apply filters ────────────────────────────────────────────────────────
    rows           = []
    selected_names = []
    n_excluded     = 0
    reason_counts  = Counter()

    for ts_name in ts_names:
        stats   = _compute_ts_stats(ts_name, alignment_data[ts_name],
                                    args.tilt_threshold)
        reasons = _apply_filters(stats, args)
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
            'n_neg_frames':   stats['n_neg_frames'],
            'n_pos_frames':   stats['n_pos_frames'],
            'alpha_deg':      _fmt(stats['alpha_deg']),
            'thickness_nm':   _fmt(stats['thickness_nm']),
            'rot_deg':        _fmt(stats['rot_deg']),
            'ref_defocus_um': _fmt(stats['ref_defocus_um']),
            'exclude_reason': '; '.join(reasons),
        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    out_path   = Path(args.output)
    fieldnames = [
        'ts_name', 'selected', 'n_frames', 'n_neg_frames', 'n_pos_frames',
        'alpha_deg', 'thickness_nm', 'rot_deg', 'ref_defocus_um', 'exclude_reason',
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

    # ── Save to project.json ─────────────────────────────────────────────────
    update_section(
        section='select_ts',
        values={
            'command':      ' '.join(sys.argv),
            'args':         args_to_dict(args),
            'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
            'analysis_dir': str(analysis_dir.resolve()),
            'csv_path':     str(out_path.resolve()),
            'n_total':      len(ts_names),
            'n_selected':   len(selected_names),
            'n_excluded':   n_excluded,
            'ts_names':     selected_names,
        },
    )
    print(f'\nSaved selection to project.json  [select_ts]')


def _fmt(v):
    if v is None:
        return ''
    return round(v, 3) if isinstance(v, float) else v
