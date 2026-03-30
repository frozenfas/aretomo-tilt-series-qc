"""
slabify — batch lamella boundary masking with slabify on AreTomo3 tomograms.

Runs `slabify` on each ts-*_Vol.mrc in an AreTomo3 output directory to generate
per-TS lamella slab masks.  Two modes are supported:

  Automatic (default)
    Fits top and bottom lamella planes by analysing local variance in the
    tomogram.  No extra input required.

  Manual
    Reads per-TS IMOD .mod or .txt control-point files from --points-dir.
    Files should be named <prefix>.mod or <prefix>.txt (e.g. ts-001.mod).

Slabify must be installed in the slabify conda environment at
/opt/miniconda3/envs/slabify/ (default), or be available on PATH.

Typical usage
-------------
  # Automatic masking of all tomograms
  aretomo3-preprocess slabify \\
      --input run002-cmd2-sart-thr80 \\
      --output slabify_masks

  # Automatic masking + write masked volumes + measure thickness
  aretomo3-preprocess slabify \\
      --input run002-cmd2-sart-thr80 \\
      --output slabify_masks \\
      --output-masked \\
      --measure

  # Manual mode with IMOD control points
  aretomo3-preprocess slabify \\
      --input run002-cmd2-sart-thr80 \\
      --output slabify_masks \\
      --points-dir lamella_points/

  # Dry run to check commands
  aretomo3-preprocess slabify \\
      --input run002-cmd2-sart-thr80 \\
      --output slabify_masks \\
      --dry-run
"""

import re
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir
from aretomo3_preprocess.shared.volume_qc import orthoslices_with_mask_b64, make_ortho_html

_SLABIFY_BIN = '/opt/miniconda3/envs/slabify/bin/slabify'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_slabify(slabify_dir=None):
    candidates = []
    if slabify_dir:
        candidates.append(str(Path(slabify_dir) / 'slabify'))
    candidates.append(_SLABIFY_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('slabify')


def _find_volumes(in_dir, vol_suffix=None):
    """Return sorted list of (prefix, vol_path) tuples."""
    if vol_suffix:
        vol_glob = f'ts-*{vol_suffix}_Vol.mrc'
    else:
        vol_glob = 'ts-*_Vol.mrc'

    vols = [v for v in sorted(in_dir.glob(vol_glob))
            if '_EVN' not in v.name and '_ODD' not in v.name]

    if not vols and not vol_suffix:
        vols = [v for v in sorted(in_dir.glob('ts-*.mrc'))
                if not any(t in v.name for t in ('_EVN', '_ODD', '_CTF'))]

    def _prefix(v):
        name = v.stem
        for tag in ('_Vol', vol_suffix or ''):
            if tag and name.endswith(tag):
                name = name[:-len(tag)]
        return name

    return [(_prefix(v), v) for v in vols]


def _find_points_file(points_dir, prefix):
    """Find a .mod or .txt control-points file for this TS prefix."""
    if points_dir is None:
        return None
    d = Path(points_dir)
    for ext in ('.mod', '.txt'):
        p = d / f'{prefix}{ext}'
        if p.exists():
            return p
    return None


def _print_cmd(cmd):
    """Print command multi-line, one flag+value per line."""
    it = iter(cmd)
    lines = ['  $ ' + next(it)]
    for tok in it:
        if tok.startswith('-'):
            lines.append('      ' + tok)
        else:
            lines[-1] += '  ' + tok
    print(' \\\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'slabify',
        help='Batch lamella boundary masking with slabify',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='AreTomo3 output directory containing ts-*_Vol.mrc files')
    inp.add_argument('--vol-suffix', default=None,
                     help='Extra suffix before _Vol.mrc (e.g. "_SART" for ts-001_SART_Vol.mrc)')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only process selected TS')
    inp.add_argument('--include', nargs='+',
                     help='Process only these TS prefixes (wildcards supported)')
    inp.add_argument('--exclude', nargs='+',
                     help='Exclude these TS prefixes (wildcards supported)')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default='slabify_masks',
                     help='Output directory for mask MRCs (ts-xxx_mask.mrc)')
    out.add_argument('--output-masked', action='store_true',
                     help='Also write masked tomogram MRCs (ts-xxx_masked.mrc)')
    out.add_argument('--measure', action='store_true',
                     help='Measure and report lamella thickness at tomogram centre')

    gen = p.add_argument_group('general options')
    gen.add_argument('--angpix', type=float, default=None, metavar='ANGST',
                     help='Pixel size in Å for output mask '
                          '(auto-read from project.json if omitted)')
    gen.add_argument('--border', type=int, default=None,
                     help='Pixels to exclude from XY border')
    gen.add_argument('--offset', type=int, default=None,
                     help='Pixels to offset from lamella boundary along Z '
                          '(positive = thicker, negative = thinner)')

    man = p.add_argument_group('manual masking (--points-dir)')
    man.add_argument('--points-dir', default=None, metavar='DIR',
                     help='Directory containing per-TS IMOD boundary control-point '
                          'files (<prefix>.mod or <prefix>.txt). '
                          'If not given, automatic mode is used.')

    auto = p.add_argument_group('automatic masking options')
    auto.add_argument('--n-samples', type=int, default=None,
                      help='Number of random points for variance analysis (default 50000)')
    auto.add_argument('--boxsize', type=int, default=None,
                      help='Box size in pixels for local variance analysis (default 32)')
    auto.add_argument('--z-min', type=int, default=None,
                      help='Minimum Z slice for variance analysis (1-indexed)')
    auto.add_argument('--z-max', type=int, default=None,
                      help='Maximum Z slice for variance analysis')
    auto.add_argument('--iterations', type=int, default=None,
                      help='Fitting iterations for boundary plane detection (default 3)')
    auto.add_argument('--simple', action='store_true',
                      help='Fit a single plane through lamella centre (assumes fixed thickness)')
    auto.add_argument('--thickness', type=int, default=None,
                      help='Lamella thickness in pixels for --simple mode '
                           '(default: half tomogram Z size)')
    auto.add_argument('--percentile', type=float, default=None,
                      help='Percentile of highest-variance locations to use for fitting '
                           '(default 95)')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report with orthogonal sections (XY, XZ, YZ) '
                         'of the tomogram with the slabify mask overlaid in transparent blue')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/slabify_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--slabify-dir', default=None,
                     help='Directory containing the slabify binary '
                          '(default: /opt/miniconda3/envs/slabify/bin/)')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without running')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    sep     = '─' * 70

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)

    if args.points_dir and not Path(args.points_dir).is_dir():
        print(f'ERROR: --points-dir {args.points_dir} not found')
        sys.exit(1)

    # Locate slabify binary
    slabify_bin = _find_slabify(args.slabify_dir)
    if not slabify_bin:
        msg = (f'slabify not found.\n'
               f'  Expected at {_SLABIFY_BIN}\n'
               f'  Or specify: --slabify-dir /path/to/slabify/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            slabify_bin = 'slabify'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    # Pixel size: only pass to slabify if explicitly given by user;
    # otherwise slabify reads it from the MRC header (correct for binning)
    angpix = args.angpix

    # Find volumes
    pairs = _find_volumes(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no tomogram volumes found in {in_dir}/')
        sys.exit(1)

    prefixes = [p for p, _ in pairs]
    vol_map  = {p: v for p, v in pairs}

    # include / exclude filtering
    if args.include:
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in inc)]
    if args.exclude:
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in exc)]

    # select-ts filter
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        orig_n   = len(prefixes)
        prefixes = [p for p in prefixes if p in selected_ts]
        n_excl   = orig_n - len(prefixes)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(prefixes)} remaining')

    if not prefixes:
        print('ERROR: no tomograms to process after filtering')
        sys.exit(1)

    mode = 'manual' if args.points_dir else 'automatic'
    print(f'Tomograms to mask: {len(prefixes)}  (mode: {mode})')
    print(sep)
    for p in prefixes[:10]:
        print(f'  {p}')
    if len(prefixes) > 10:
        print(f'  ... ({len(prefixes) - 10} more)')
    print(sep)

    out_dir.mkdir(parents=True, exist_ok=True)

    do_qc    = getattr(args, 'analyse', False)
    qc_entries = []

    ok, failed, skipped = [], [], []

    for i, prefix in enumerate(prefixes):
        print(f'\n[{i+1}/{len(prefixes)}] {prefix}')

        tomo     = vol_map[prefix]
        mask_out = out_dir / f'{prefix}_mask.mrc'

        cmd = [
            slabify_bin,
            '--input',  str(tomo),
            '--output', str(mask_out),
        ]

        if args.output_masked:
            masked_out = out_dir / f'{prefix}_masked.mrc'
            cmd += ['--output-masked', str(masked_out)]

        if angpix is not None:
            cmd += ['--angpix', str(angpix)]
        if args.border is not None:
            cmd += ['--border', str(args.border)]
        if args.offset is not None:
            cmd += ['--offset', str(args.offset)]
        if args.measure:
            cmd += ['--measure']

        if args.points_dir:
            # Manual mode
            points_file = _find_points_file(args.points_dir, prefix)
            if points_file is None:
                print(f'  WARNING: no .mod or .txt found for {prefix} in '
                      f'{args.points_dir} — skipping')
                skipped.append(prefix)
                continue
            cmd += ['--points', str(points_file)]
        else:
            # Automatic mode options
            if args.n_samples is not None:
                cmd += ['--n-samples', str(args.n_samples)]
            if args.boxsize is not None:
                cmd += ['--boxsize', str(args.boxsize)]
            if args.z_min is not None:
                cmd += ['--z-min', str(args.z_min)]
            if args.z_max is not None:
                cmd += ['--z-max', str(args.z_max)]
            if args.iterations is not None:
                cmd += ['--iterations', str(args.iterations)]
            if args.simple:
                cmd += ['--simple']
            if args.thickness is not None:
                cmd += ['--thickness', str(args.thickness)]
            if args.percentile is not None:
                cmd += ['--percentile', str(args.percentile)]

        _print_cmd(cmd)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(prefix)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: slabify exited with code {ret.returncode}')
            failed.append(prefix)
            continue

        ok.append(prefix)
        print(f'  → {mask_out}')
        if args.output_masked:
            print(f'  → {out_dir / f"{prefix}_masked.mrc"}')

        # QC
        if do_qc:
            img_b64 = None
            if mask_out.exists():
                img_b64 = orthoslices_with_mask_b64(tomo, mask_out)
            qc_entries.append({
                'ts_name':   prefix,
                'img_b64':   img_b64,
                'tomo_path': str(tomo),
                'mask_path': str(mask_out),
                'metadata': {
                    'mode':   mode,
                    'angpix': f'{angpix} Å' if angpix else 'unknown',
                },
            })

    # Summary
    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed'
          + (f', {len(skipped)} skipped (no points file)' if skipped else '') + '.')
    if failed:
        print(f'Failed:  {", ".join(failed)}')
    if skipped:
        print(f'Skipped: {", ".join(skipped)}')
    if not args.dry_run and ok:
        print(f'Masks:   {out_dir}/')

    # QC report
    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'slabify_qc.html')
        make_ortho_html(
            entries  = qc_entries,
            out_path = html_path,
            title    = 'slabify QC',
            command  = ' '.join(sys.argv),
        )

    if args.dry_run:
        return

    update_section(
        section='slabify',
        values={
            'command':     ' '.join(sys.argv),
            'args':        args_to_dict(args),
            'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
            'n_processed': len(ok),
            'failed':      failed,
            'skipped':     skipped,
            'input_dir':   str(in_dir),
            'output_dir':  str(out_dir),
            'mode':        mode,
        },
        backup_dir=out_dir,
    )
