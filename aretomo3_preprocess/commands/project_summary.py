"""
project-summary — one-page colored terminal dashboard of project.json state.

Reads whatever this project's aretomo3_project.json + (if analyse has run)
alignment_data.json already have on disk and prints a concise overview: mdoc/
rename counts, key paths, CTF/matching params, external tool paths, every
recorded analysis run and where its report lives, and outlier-trimmed
thickness/defocus/overlap distributions with small terminal histograms.

Purely a read-only view over data other commands already recorded -- this
command computes nothing new and writes nothing back to project.json.
"""

import sys
import json
import datetime
from pathlib import Path
from collections import Counter

import numpy as np

from aretomo3_preprocess.shared.project_json import load as load_project

# ── ANSI colour helpers (no external dep -- hand-rolled) ────────────────────
_RESET, _BOLD, _DIM = '\033[0m', '\033[1m', '\033[2m'
_CYAN, _GREEN, _YELLOW, _RED, _BLUE, _MAGENTA = (
    '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[34m', '\033[35m',
)


def _c(text, *codes, use_color=True):
    if not use_color:
        return str(text)
    return ''.join(codes) + str(text) + _RESET


def _hr(width, use_color, ch='─'):
    print(_c(ch * width, _DIM, use_color=use_color))


def _kv(label, value, width=26, use_color=True, value_color=None):
    label_s = str(label)
    # Always at least one space before the value, even when label overflows
    # width -- f'{label:<{width}}' silently glues value onto label with no
    # separator at all once label is already >= width chars.
    pad = ' ' * max(1, width - len(label_s))
    val_s = _c(value, value_color, use_color=use_color) if value_color else str(value)
    print(f'  {_c(label_s, _DIM, use_color=use_color)}{pad}{val_s}')


def _section(title, use_color, width=78):
    print()
    print(_c(f' {title} ', _BOLD, _CYAN, use_color=use_color).center(
        width + (len(_BOLD) + len(_CYAN) + len(_RESET) if use_color else 0), '─'))


def _percentile_range(vals, lo_pct=2, hi_pct=98):
    clean = [v for v in vals if v is not None]
    if not clean:
        return None, None, []
    if len(clean) < 5:
        return min(clean), max(clean), clean
    lo, hi = float(np.percentile(clean, lo_pct)), float(np.percentile(clean, hi_pct))
    trimmed = [v for v in clean if lo <= v <= hi]
    return lo, hi, trimmed


_BLOCKS = ' ▁▂▃▄▅▆▇█'


def _ascii_hist(vals, width=50, height=6, use_color=True, unit=''):
    """Small terminal histogram: `height` rows of block characters, one
    column per bin across `width` columns."""
    clean = sorted(v for v in vals if v is not None)
    if len(clean) < 2:
        print(_c('    (not enough data)', _DIM, use_color=use_color))
        return
    lo, hi = clean[0], clean[-1]
    if lo == hi:
        hi = lo + 1e-9
    n_bins = min(width, max(10, len(clean) // 2))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(clean, bins=edges)
    max_count = counts.max() if counts.max() > 0 else 1

    # Render height rows top-down using sub-block resolution per column.
    for row in range(height, 0, -1):
        line = []
        for cnt in counts:
            level = cnt / max_count * height
            filled = level - (row - 1)
            if filled >= 1:
                line.append(_BLOCKS[-1])
            elif filled > 0:
                line.append(_BLOCKS[int(filled * (len(_BLOCKS) - 1))])
            else:
                line.append(' ')
        print('    ' + _c(''.join(line), _GREEN, use_color=use_color))
    print(f'    {lo:.2f}{unit}' + ' ' * max(1, n_bins - len(f'{lo:.2f}{unit}') - len(f'{hi:.2f}{unit}')) + f'{hi:.2f}{unit}')


def _median(vals):
    clean = [v for v in vals if v is not None]
    return float(np.median(clean)) if clean else None


def add_parser(subparsers):
    p = subparsers.add_parser(
        'project-summary',
        help='Print a one-page colored terminal summary of this project',
        description=__doc__,
    )
    p.add_argument('--no-color', action='store_true',
                   help='Disable ANSI colors (auto-disabled anyway when not a terminal)')
    p.set_defaults(func=run)
    return p


def run(args):
    proj = load_project()
    use_color = (not args.no_color) and sys.stdout.isatty()
    W = 78

    project_dir = Path.cwd()
    print()
    print(_c(f'  AreTomo3-Preprocess project summary', _BOLD, use_color=use_color))
    print(_c(f'  {project_dir}', _DIM, use_color=use_color))
    last_updated = proj.get('project', {}).get('last_updated', '?')
    print(_c(f'  last updated: {last_updated}', _DIM, use_color=use_color))

    # ── mdoc / renaming ───────────────────────────────────────────────────
    _section('mdoc & renaming', use_color, W)
    mdoc = proj.get('mdoc_data', {})
    per_ts = mdoc.get('per_ts', {})
    n_mdoc = len(per_ts)
    rename = proj.get('rename_ts', {})
    n_renamed = len(rename.get('lookup', {}))
    angpix_vals = [v.get('angpix') for v in per_ts.values() if v.get('angpix') is not None]
    voltages = [v.get('acquisition', {}).get('voltage') for v in per_ts.values()
               if v.get('acquisition', {}).get('voltage') is not None]
    angpix_mode = Counter(angpix_vals).most_common(1)[0][0] if angpix_vals else None
    voltage_mode = Counter(voltages).most_common(1)[0][0] if voltages else None
    calibrated = proj.get('calibrated_apix', {})

    _kv('mdoc files cached', n_mdoc, use_color=use_color)
    _kv('ts-XXX renamed', n_renamed, use_color=use_color)
    _kv('pixel size (mdoc mode)', f'{angpix_mode} Å/px' if angpix_mode else 'n/a', use_color=use_color)
    if calibrated.get('value') is not None:
        _kv('pixel size (calibrated)', f'{calibrated["value"]} Å/px  (from {calibrated.get("source", "?")})',
            use_color=use_color, value_color=_GREEN)
    _kv('voltage', f'{voltage_mode:.0f} kV' if voltage_mode else 'n/a', use_color=use_color)

    # ── Paths ─────────────────────────────────────────────────────────────
    _section('paths', use_color, W)
    stacks = proj.get('input_stacks', {})
    _kv('cmd0 output dir', stacks.get('cmd0_outdir', 'n/a'), use_color=use_color)
    _kv('mrc stacks registered', stacks.get('n_stacks', 'n/a'), use_color=use_color)
    tlt_dir = stacks.get('tlt_dir')
    _kv('TLT dir', tlt_dir or 'n/a', use_color=use_color)

    tool_paths = proj.get('tool_paths', {})
    for tool in ('aretomo3', 'pytom', 'imod', 'gapstop'):
        _kv(f'{tool} path', tool_paths.get(tool, _c('not recorded', _DIM, use_color=use_color)),
            use_color=use_color)

    # ── Matching / CTF params ────────────────────────────────────────────
    run_params = proj.get('run_aretomo3_params', {})
    if run_params:
        _section('run-aretomo3 params (self-consistency baseline)', use_color, W)
        _kv('kv', run_params.get('kv', 'n/a'), use_color=use_color)
        _kv('cs', run_params.get('cs', 'n/a'), use_color=use_color)
        _kv('amp_contrast', run_params.get('amp_contrast', 'n/a'), use_color=use_color)
        _kv('apix', run_params.get('apix', 'n/a'), use_color=use_color)

    # ── Handedness / lamellae ────────────────────────────────────────────
    handedness = proj.get('handedness')
    lamellae = proj.get('lamella_assignments', {})
    if handedness or lamellae:
        _section('handedness & lamellae', use_color, W)
        if handedness:
            mirror = handedness.get('mirror')
            _kv('handedness (mirror needed?)',
                ('YES — use --mirror' if mirror else 'no'),
                use_color=use_color, value_color=(_YELLOW if mirror else _GREEN))
            _kv('  determined from', f'{handedness.get("particle", "?")}, '
                f'{len(handedness.get("per_ts", {}))} TS, {handedness.get("timestamp", "?")}',
                use_color=use_color)
        if lamellae:
            n_lam = len(set(lamellae.get('positions', {}).values()))
            _kv('lamellae', f'{n_lam}  ({lamellae.get("n_ts", "?")} TS assigned)', use_color=use_color)

    # ── Analysis runs (jobs run + report locations) ──────────────────────
    runs = proj.get('analysis_runs', [])
    _section('analysis runs & reports', use_color, W)
    landing = project_dir / 'analysis_start.html'
    if landing.exists():
        _kv('report index', str(landing), use_color=use_color, value_color=_CYAN)
    if runs:
        for r in runs:
            _kv(f'  [{r.get("kind", "?")}] {r.get("label", "?")}',
                f'{r.get("output_dir", "?")}  ({r.get("timestamp", "?")})',
                use_color=use_color)
    else:
        print(_c('    (none recorded yet)', _DIM, use_color=use_color))

    # ── Alignment stats (from analyse's alignment_data.json, if present) ──
    analyse_sec = proj.get('analyse', {})
    align_json = None
    if analyse_sec.get('output_dir'):
        candidate = Path(analyse_sec['output_dir'])
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        cand_file = candidate / 'alignment_data.json'
        if cand_file.exists():
            align_json = cand_file

    if align_json is not None:
        with open(align_json) as fh:
            all_ts = json.load(fh)

        rots, thickness_nm, overlaps, defoci = [], [], [], []
        for ts, d in all_ts.items():
            frames = d.get('frames', [])
            if frames:
                rots.append(frames[0].get('rot'))
            if d.get('thickness_nm') is not None and d['thickness_nm'] < 5000:
                thickness_nm.append(d['thickness_nm'])
            for f in frames:
                if f.get('overlap_pct') is not None:
                    overlaps.append(f['overlap_pct'])
                if f.get('mean_defocus_um') is not None:
                    defoci.append(f['mean_defocus_um'])

        _section(f'alignment stats — {len(all_ts)} TS ({align_json})', use_color, W)
        med_rot = _median(rots)
        _kv('median tilt axis (ROT)', f'{med_rot:.2f}°' if med_rot is not None else 'n/a',
            use_color=use_color)
        med_ovl = _median(overlaps)
        _kv('median frame overlap', f'{med_ovl:.1f}%' if med_ovl is not None else 'n/a',
            use_color=use_color, value_color=(_GREEN if (med_ovl or 0) > 80 else _YELLOW))

        thk_lo, thk_hi, thk_trim = _percentile_range(thickness_nm)
        if thk_trim:
            print(f'\n  {_c("thickness (nm), 2nd-98th pctile:", _DIM, use_color=use_color)} '
                  f'{thk_lo:.0f} – {thk_hi:.0f}  (median {_median(thk_trim):.0f})')
            _ascii_hist(thk_trim, use_color=use_color, unit='nm')

        def_lo, def_hi, def_trim = _percentile_range(defoci)
        if def_trim:
            print(f'\n  {_c("defocus (µm), 2nd-98th pctile:", _DIM, use_color=use_color)} '
                  f'{def_lo:.2f} – {def_hi:.2f}  (median {_median(def_trim):.2f})')
            _ascii_hist(def_trim, use_color=use_color, unit='µm')
    else:
        _section('alignment stats', use_color, W)
        print(_c('    No alignment_data.json found (run analyse first).', _DIM, use_color=use_color))

    print()
