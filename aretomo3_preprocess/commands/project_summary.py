"""
project-summary — one-page colored terminal dashboard of project.json state.

Reads only aretomo3_project.json for this project (no other file on disk is
opened) and prints a concise overview: mdoc/rename counts with a renaming
example, calibrated vs. raw mdoc pixel size, full CTF/optics params, gain
transform, external tool paths, handedness, every recorded analysis run and
where its report lives, and outlier-trimmed thickness/defocus/overlap/tilt-
axis distributions (with optional terminal histograms via --plot).

Purely a read-only view over data other commands already recorded -- this
command computes nothing new and writes nothing back to project.json. The
per-TS QC distributions come from `analyse`'s own `per_ts_qc`/`qc_summary`
project.json fields (persisted by analyse.py itself) rather than opening
that run's alignment_data.json directly, so this command's only I/O is
project.json.

The actual project.json reading/computation lives in
`shared/project_summary_data.py`'s `gather()` -- shared with
`shared/landing_page.py`, which renders the same data (minus histograms) as
an HTML panel on the project's `analysis_start.html` landing page, so the
CLI and the "first page" of the HTML report never drift out of sync with
each other. This module only adds the ANSI/terminal rendering on top.
"""

import sys
from pathlib import Path

import numpy as np

from aretomo3_preprocess.shared.project_json import load as load_project, SCHEMA_VERSION
from aretomo3_preprocess.shared.project_summary_data import gather

# ── ANSI colour helpers (no external dep -- hand-rolled) ────────────────────
_RESET, _BOLD, _DIM = '\033[0m', '\033[1m', '\033[2m'
_CYAN, _GREEN, _YELLOW, _RED, _BLUE, _MAGENTA = (
    '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[34m', '\033[35m',
)


def _c(text, *codes, use_color=True):
    if not use_color:
        return str(text)
    return ''.join(codes) + str(text) + _RESET


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


def _note(text, use_color, width=78):
    print(_c(f'    {text}', _DIM, use_color=use_color))


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


def add_parser(subparsers):
    p = subparsers.add_parser(
        'project-summary',
        help='Print a one-page colored terminal summary of this project',
        description=__doc__,
    )
    p.add_argument('--no-color', action='store_true',
                   help='Disable ANSI colors (auto-disabled anyway when not a terminal)')
    p.add_argument('--plot', action='store_true',
                   help='Also print terminal histograms for thickness/defocus/'
                        'overlap/tilt-axis distributions (requires analyse to '
                        'have been re-run since per_ts_qc was added; see the '
                        'QC distributions section otherwise)')
    p.set_defaults(func=run)
    return p


def run(args):
    proj = load_project()
    use_color = (not args.no_color) and sys.stdout.isatty()
    W = 78
    d = gather(proj)

    project_dir = Path.cwd()
    print()
    print(_c(f'  AreTomo3-Preprocess project summary', _BOLD, use_color=use_color))
    print(_c(f'  {project_dir}', _DIM, use_color=use_color))
    print(_c(f'  last updated: {d["last_updated"]}', _DIM, use_color=use_color))
    if d['schema_version'] and d['schema_version'] < SCHEMA_VERSION:
        print(_c(f'  NOTE: project.json schema v{d["schema_version"]} (this tool expects '
                  f'v{SCHEMA_VERSION}) -- some fields below may be missing until the '
                  f'relevant command is re-run.', _YELLOW, use_color=use_color))

    # Everything below is read from aretomo3_project.json only -- no other
    # file on disk is opened. Any value not present there is reported as
    # explicitly missing (with a reason/fix), never silently skipped.

    # ── mdoc / renaming ───────────────────────────────────────────────────
    _section('mdoc & renaming', use_color, W)
    _kv('mdoc files cached', d['n_mdoc'], use_color=use_color)
    _kv('ts-XXX renamed', d['n_renamed'], use_color=use_color)
    if d['renaming_example']:
        ts_stem, orig_stem = d['renaming_example']
        _kv('renaming example', f'{ts_stem}  →  {orig_stem}', use_color=use_color)

    cal = d['calibrated']
    cal_value = cal.get('value')
    angpix_mdoc = d['angpix_mdoc']
    if cal_value is not None:
        bracket = f'  ({angpix_mdoc} from mdoc)' if (angpix_mdoc is not None
                  and abs(angpix_mdoc - cal_value) > 1e-6) else ''
        _kv('pixel size', f'{cal_value} Å/px{bracket}', use_color=use_color, value_color=_GREEN)
        _kv('  calibrated via', cal.get('source', '?'), use_color=use_color)
    elif angpix_mdoc is not None:
        _kv('pixel size', f'{angpix_mdoc} Å/px  (raw mdoc, not yet calibrated)',
            use_color=use_color, value_color=_YELLOW)
    else:
        _kv('pixel size', 'n/a', use_color=use_color)

    w_mode, h_mode = d['image_dims']
    voltage_mode = d['voltage_mode']
    _kv('voltage', f'{voltage_mode:.0f} kV' if voltage_mode else 'n/a', use_color=use_color)
    _kv('image dimensions', f'{w_mode} × {h_mode} px' if (w_mode and h_mode) else 'n/a',
        use_color=use_color)
    if d['frames_per_tilt'] is not None:
        _kv('frames per tilt', d['frames_per_tilt'], use_color=use_color)
    if d['tilt_range']:
        lo, hi, n_t, step = d['tilt_range']
        _kv(f'tilt range (from {d["ts_key"]})',
            f'{lo:+.1f}° → {hi:+.1f}°  ({n_t} tilts, ~{step:.2f}° step)', use_color=use_color)

    dose = d['dose']
    if dose['status'] == 'ok':
        _kv(f'dose (--fm-dose, cmd=0, {dose["timestamp"]})',
            f'{dose["per_frame"]:.3f} e⁻/Å² per frame  |  {dose["per_tilt"]:.2f} e⁻/Å² per tilt  |  '
            f'~{dose["total"]:.0f} e⁻/Å² total', use_color=use_color)
    elif dose['status'] == 'stale_cmd':
        _kv('dose', f'--fm-dose was {dose["per_frame"]} e⁻/Å²/frame on the last '
                    f'run-aretomo3 invocation, but that was cmd={dose["cmd"]}, '
                    f'not cmd=0 -- project.json only keeps the most recent '
                    f'run-aretomo3 call, so this may not be the cmd0 value.',
            use_color=use_color, value_color=_YELLOW)
    else:
        _kv('dose', 'not available -- no run-aretomo3 --fm-dose recorded '
                    'in project.json yet', use_color=use_color, value_color=_DIM)

    # ── CTF / optics ──────────────────────────────────────────────────────
    if d['ctf']:
        run_params = d['ctf']
        _section('CTF / optics params (run-aretomo3 self-consistency baseline)', use_color, W)
        _kv('voltage', f'{run_params.get("kv", "n/a")} kV', use_color=use_color)
        _kv('spherical aberration (Cs)', f'{run_params.get("cs", "n/a")} mm', use_color=use_color)
        _kv('amplitude contrast', run_params.get('amp_contrast', 'n/a'), use_color=use_color)
        _kv('pixel size used', f'{run_params.get("apix", "n/a")} Å/px', use_color=use_color)
        _kv('  from cmd', run_params.get('cmd', '?'), use_color=use_color)

    # ── Gain transform ────────────────────────────────────────────────────
    if d['gain_check']:
        gc = d['gain_check']
        _section('gain transform', use_color, W)
        _kv('best transform', gc.get('best_transform', 'n/a'), use_color=use_color, value_color=_GREEN)
        _kv('AreTomo3 flags', f'-RotGain {gc.get("aretomo3_rot_gain", "?")} '
            f'-FlipGain {gc.get("aretomo3_flip_gain", "?")}', use_color=use_color)
        gain_file = gc.get('args', {}).get('gain')
        if gain_file:
            _kv('gain file', gain_file, use_color=use_color)

    # ── Paths ─────────────────────────────────────────────────────────────
    _section('paths', use_color, W)
    paths = d['paths']
    _kv('cmd0 output dir', paths['cmd0_outdir'] or 'n/a', use_color=use_color)
    _kv('mrc stacks registered', paths['n_stacks'] if paths['n_stacks'] is not None else 'n/a',
        use_color=use_color)
    _kv('TLT dir', paths['tlt_dir'] or 'n/a', use_color=use_color)
    if paths['tlt_dir']:
        _note('only refreshed by "run-aretomo3 --cmd 0" -- a later --cmd 1/2 run '
              'does NOT update this, so it may point at an older cmd0 pass than '
              'whatever alignment run you last analysed.', use_color, W)

    for tool in ('aretomo3', 'pytom', 'imod', 'gapstop'):
        _kv(f'{tool} path', paths['tool_paths'].get(tool, _c('not recorded', _DIM, use_color=use_color)),
            use_color=use_color)

    # ── Handedness / lamellae ────────────────────────────────────────────
    _section('handedness & lamellae', use_color, W)
    handedness = d['handedness']
    if handedness:
        mirror = handedness.get('mirror')
        _kv('physical (mirror) handedness',
            ('YES — use --mirror' if mirror else 'no mirroring needed'),
            use_color=use_color, value_color=(_YELLOW if mirror else _GREEN))
        _kv('  determined from', f'{handedness.get("particle", "?")}, '
            f'{len(handedness.get("per_ts", {}))} TS, {handedness.get("timestamp", "?")}',
            use_color=use_color)
    else:
        _kv('physical (mirror) handedness', 'not determined yet -- run '
            'pytom-ribo-auto --check-handedness', use_color=use_color, value_color=_DIM)
    dg = d['defocusgrad']
    if dg:
        consensus = dg.get('consensus', '?')
        n_ts = len(dg.get('per_ts', {}))
        is_warn = 'inconsist' in str(consensus) or 'inconclus' in str(consensus)
        _kv('defocus handedness', consensus, use_color=use_color,
            value_color=(_YELLOW if is_warn else _GREEN))
        _kv('  determined from', f'defocusgrad, {n_ts} TS', use_color=use_color)
    else:
        _kv('defocus handedness', 'not checked yet -- run defocusgrad',
            use_color=use_color, value_color=_DIM)
    if d['lamellae']:
        n_lam, n_ts_assigned = d['lamellae']
        _kv('lamellae', f'{n_lam}  ({n_ts_assigned} TS assigned)', use_color=use_color)

    # ── Analysis runs (jobs run + report locations) ──────────────────────
    _section('analysis runs & reports', use_color, W)
    landing = project_dir / 'analysis_start.html'
    if landing.exists():
        _kv('report index', str(landing), use_color=use_color, value_color=_CYAN)
    if d['analysis_runs']:
        for r in d['analysis_runs']:
            _kv(f'  [{r.get("kind", "?")}] {r.get("label", "?")}',
                f'{r.get("output_dir", "?")}  ({r.get("timestamp", "?")})',
                use_color=use_color)
    else:
        print(_c('    (none recorded yet)', _DIM, use_color=use_color))

    # ── QC distributions (from analyse's own project.json fields) ────────
    analyse_sec = d['analyse_sec']
    qc_summary  = d['qc_summary']
    per_ts_qc   = d['per_ts_qc']

    if qc_summary:
        n_ts = analyse_sec.get('n_tilt_series', len(per_ts_qc or []))
        _section(f'QC distributions — {n_ts} TS, 2nd-98th pctile trimmed '
                 f'(from {analyse_sec.get("output_dir", "?")})', use_color, W)
        _note(f'from the last "analyse" run of any cmd stage (timestamp '
              f'{analyse_sec.get("timestamp", "?")}) -- independent of the TLT '
              f'dir above, which only tracks the last cmd0 run.', use_color, W)
        print()

        def _range_line(key, label, unit):
            r = qc_summary.get(key)
            if not r:
                return
            print(f'  {_c(label + ":", _DIM, use_color=use_color)} '
                  f'{r["lo"]:.2f} – {r["hi"]:.2f} {unit}  (median {r["median"]:.2f})')
            if args.plot and per_ts_qc:
                # Histogram over the same 2nd-98th pctile trimmed range the
                # text line above reports, not the raw per-TS values -- a
                # single genuine outlier (e.g. a mis-parsed thickness) would
                # otherwise blow the bin range out far past what's shown as
                # the dataset's actual range, making the histogram and the
                # text line above it visually contradict each other.
                trimmed = [v for v in (e.get(key) for e in per_ts_qc)
                          if v is not None and r['lo'] <= v <= r['hi']]
                _ascii_hist(trimmed, use_color=use_color, unit=unit)

        _range_line('tilt_axis_deg',  'median tilt axis (ROT)', '°')
        _range_line('mean_overlap',   'mean frame overlap',     '%')
        _range_line('thickness_nm',   'thickness',              'nm')
        _range_line('avg_defocus_um', 'defocus',                'µm')
    elif analyse_sec:
        _section('QC distributions', use_color, W)
        _note('This project\'s project.json was last written by an older '
              'analyse.py that did not persist per-TS QC data -- re-run '
              '"analyse" (--reuse-plots is fine) to populate this section.',
              use_color, W)
    else:
        _section('QC distributions', use_color, W)
        _note('No analyse run recorded yet.', use_color, W)

    print()
