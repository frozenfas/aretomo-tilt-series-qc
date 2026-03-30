"""
pytom-extract — extract template matching candidates from pytom-match output.

Runs pytom_extract_candidates.py on each *_job.json in the pytom-match
output directory, then optionally converts the resulting STAR files to
IMOD .mod point models for visualisation in 3dmod.

IMOD conversion adapted from rln2mod by Phaips
(https://github.com/Phaips/rln2mod). Original script reads RELION5 STAR
centred coordinates and calls IMOD point2model; here integrated as a
sub-step with auto-detected tomogram dimensions from the job JSON.

Typical usage
-------------
  # Extract candidates from a pytom-match run
  aretomo3-preprocess pytom-extract \\
      --input pytom_match_test \\
      --n-particles 2000 \\
      --particle-diameter 300 \\
      --tophat-filter

  # Extract and convert to IMOD .mod files for 3dmod visualisation
  aretomo3-preprocess pytom-extract \\
      --input pytom_match_test \\
      --n-particles 2000 \\
      --particle-diameter 300 \\
      --tophat-filter \\
      --imod
"""

import json
import re
import shutil
import struct
import sys
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts

_PYTOM_EXTRACT_BIN = '/opt/miniconda3/envs/pytom_tm/bin/pytom_extract_candidates.py'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_pytom_extract(pytom_dir=None):
    candidates = []
    if pytom_dir:
        candidates.append(str(Path(pytom_dir) / 'pytom_extract_candidates.py'))
    candidates.append(_PYTOM_EXTRACT_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('pytom_extract_candidates.py')


def _mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC header without mrcfile dependency."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)  # nx, ny, nz


def _find_job_jsons(input_dir, selected_ts=None):
    """Return sorted list of (ts_name, job_json_path) tuples."""
    jobs = []
    for ts_dir in sorted(Path(input_dir).iterdir()):
        if not ts_dir.is_dir():
            continue
        ts_name = ts_dir.name
        if selected_ts is not None and ts_name not in selected_ts:
            continue
        matches = sorted(ts_dir.glob('*_job.json'))
        if matches:
            jobs.append((ts_name, matches[0]))
    return jobs


def _star_to_mod(star_path, job_json_path, mod_dir):
    """
    Convert a RELION5 particles STAR file to an IMOD .mod point model.

    Adapted from rln2mod by Phaips (https://github.com/Phaips/rln2mod).
    Reads rlnCenteredCoordinate{X,Y,Z}Angst and rlnTomoTiltSeriesPixelSize,
    converts to pixel coordinates, writes a .txt, then calls point2model.

    Tomogram dimensions are read from the MRC referenced in the job JSON.
    """
    try:
        import starfile
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
    except ImportError:
        print('  WARNING: starfile not installed in current env — skipping IMOD conversion')
        print('           conda run -n pytom_tm pip install starfile')
        return False

    # Get tomogram dimensions from the job JSON
    with open(job_json_path) as fh:
        job = json.load(fh)
    tomo_path = job.get('tomogram') or job.get('tomogram_path') or job.get('volume_path')
    if not tomo_path or not Path(tomo_path).exists():
        print(f'  WARNING: cannot find tomogram path in {job_json_path.name} — skipping IMOD conversion')
        return False

    nx, ny, nz = _mrc_dims(tomo_path)

    # Read STAR
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        data = starfile.read(str(star_path))
    df = data[next(iter(data))] if isinstance(data, dict) else data

    required = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
                 'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    missing = required - set(df.columns)
    if missing:
        print(f'  WARNING: STAR file missing columns {missing} — '
              f're-run extraction with --relion5-compat')
        return False

    px = df['rlnTomoTiltSeriesPixelSize']
    xs = df['rlnCenteredCoordinateXAngst'] / px + nx / 2
    ys = df['rlnCenteredCoordinateYAngst'] / px + ny / 2
    zs = df['rlnCenteredCoordinateZAngst'] / px + nz / 2

    mod_dir.mkdir(parents=True, exist_ok=True)
    stem  = star_path.stem
    txt   = mod_dir / f'{stem}.txt'
    mod   = mod_dir / f'{stem}.mod'

    with open(txt, 'w') as fh:
        for x, y, z in zip(xs, ys, zs):
            fh.write(f'{x:.6f} {y:.6f} {z:.6f}\n')

    point2model = shutil.which('point2model')
    if not point2model:
        print('  WARNING: point2model not found on PATH — load IMOD first')
        txt.unlink(missing_ok=True)
        return False

    ret = subprocess.run([point2model, str(txt), str(mod)], capture_output=True)
    if ret.returncode != 0:
        print(f'  WARNING: point2model failed: {ret.stderr.decode().strip()}')
        return False

    txt.unlink(missing_ok=True)
    print(f'  → {mod}  ({len(df)} particles)')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'pytom-extract',
        help='Extract template matching candidates from pytom-match output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='pytom-match output directory (contains per-TS subdirectories)')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only process selected TS')
    inp.add_argument('--include', nargs='+',
                     help='Process only these TS names (wildcards supported; '
                          'comma-separated or space-separated)')
    inp.add_argument('--exclude', nargs='+',
                     help='Exclude these TS names (wildcards supported)')

    ext = p.add_argument_group('extraction (pytom_extract_candidates.py)')
    ext.add_argument('--n-particles', '-n', type=int, required=True,
                     help='Maximum candidates to extract per tomogram')
    ext.add_argument('--particle-diameter', type=float, default=None,
                     help='Particle diameter in Å — sets exclusion radius between picks '
                          '(min peak-to-peak distance = diameter / 2)')
    ext.add_argument('--tophat-filter', action='store_true',
                     help='Apply tophat filter to flatten uneven background '
                          'before picking (recommended)')
    ext.add_argument('--tophat-bins', type=int, default=None,
                     help='Number of bins for tophat filter')
    ext.add_argument('--cut-off', type=float, default=None,
                     help='Override automated LCCmax cut-off; set to 0 to extract '
                          'all n-particles regardless of score')
    ext.add_argument('--n-false-positives', type=int, default=None,
                     help='Number of false positives for cut-off estimation '
                          '(default 1; increase to improve recall at cost of FP rate)')
    ext.add_argument('--relion5-compat', action='store_true',
                     help='Write RELION5-compatible STAR files '
                          '(required for --imod conversion)')
    ext.add_argument('--log', choices=['info', 'debug'], default=None,
                     help='Logging level for pytom_extract_candidates.py')

    imod = p.add_argument_group('IMOD visualisation')
    imod.add_argument('--imod', action='store_true',
                      help='Convert extracted STAR files to IMOD .mod point models '
                           '(requires --relion5-compat and IMOD point2model on PATH). '
                           'Adapted from rln2mod (https://github.com/Phaips/rln2mod).')
    imod.add_argument('--imod-dir', default=None,
                      help='Directory for .mod files (default: <input>/mod/)')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report showing particle picks overlaid '
                         'on the central slab of each tomogram')
    qc.add_argument('--analyse-thickness', type=float, default=300.0, metavar='ANGST',
                    help='Slab thickness in Å for QC report (default: 300 Å)')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <input>/pytom_extract_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--pytom-dir', default=None,
                     help='Directory containing pytom_extract_candidates.py '
                          '(default: /opt/miniconda3/envs/pytom_tm/bin/)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without running')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input).resolve()
    sep     = '─' * 70

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    if args.imod and not args.relion5_compat:
        print('WARNING: --imod requires --relion5-compat for correct STAR column names. '
              'Adding --relion5-compat automatically.')
        args.relion5_compat = True

    extract_bin = _find_pytom_extract(args.pytom_dir)
    if not extract_bin:
        msg = (f'pytom_extract_candidates.py not found.\n'
               f'  Expected at {_PYTOM_EXTRACT_BIN}\n'
               f'  Or specify: --pytom-dir /path/to/pytom_tm/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            extract_bin = 'pytom_extract_candidates.py'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    mod_dir = Path(args.imod_dir).resolve() if args.imod_dir else in_dir / 'mod'

    # Find job JSONs
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    jobs = _find_job_jsons(in_dir, selected_ts)

    # include / exclude filtering
    if getattr(args, 'include', None):
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        jobs = [(n, j) for n, j in jobs
                if any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in inc)]
    if getattr(args, 'exclude', None):
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        jobs = [(n, j) for n, j in jobs
                if not any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in exc)]

    if not jobs:
        print(f'ERROR: no *_job.json files found in {in_dir}/')
        sys.exit(1)

    print(f'Tomograms to extract: {len(jobs)}')
    print(sep)
    for ts_name, _ in jobs[:10]:
        print(f'  {ts_name}')
    if len(jobs) > 10:
        print(f'  ... ({len(jobs) - 10} more)')
    print(sep)

    # ── QC setup ──────────────────────────────────────────────────────────────
    do_qc    = getattr(args, 'analyse', False)
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    if do_qc:
        try:
            import warnings
            import starfile as _starfile
            from aretomo3_preprocess.shared.volume_qc import (
                slab_with_picks_b64, make_picks_html,
            )
        except ImportError as e:
            print(f'WARNING: --analyse requires starfile and matplotlib ({e}); skipping report')
            do_qc = False

    ok, failed = [], []

    for i, (ts_name, job_json) in enumerate(jobs):
        print(f'\n[{i+1}/{len(jobs)}] {ts_name}')

        # Build extraction command
        cmd = [
            extract_bin,
            '-j', str(job_json),
            '-n', str(args.n_particles),
        ]
        if args.particle_diameter is not None:
            cmd += ['--particle-diameter', str(args.particle_diameter)]
        if args.tophat_filter:
            cmd += ['--tophat-filter']
        if args.tophat_bins is not None:
            cmd += ['--tophat-bins', str(args.tophat_bins)]
        if args.cut_off is not None:
            cmd += ['--cut-off', str(args.cut_off)]
        if args.n_false_positives is not None:
            cmd += ['--number-of-false-positives', str(args.n_false_positives)]
        if args.relion5_compat:
            cmd += ['--relion5-compat']
        if args.log:
            cmd += ['--log', args.log]

        # Print command
        it = iter(cmd)
        lines = ['  $ ' + next(it)]
        for tok in it:
            if tok.startswith('-'):
                lines.append('      ' + tok)
            else:
                lines[-1] += '  ' + tok
        print(' \\\n'.join(lines))

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(ts_name)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: extraction exited with code {ret.returncode}')
            failed.append(ts_name)
            continue

        ok.append(ts_name)

        # IMOD conversion
        if args.imod:
            star_files = sorted(job_json.parent.glob('*_particles.star'))
            if not star_files:
                print(f'  WARNING: no *_particles.star found — skipping IMOD conversion')
            else:
                for star in star_files:
                    _star_to_mod(star, job_json, mod_dir)

        # QC: render central slab with picks
        if do_qc:
            with open(job_json) as fh:
                job = json.load(fh)
            tomo_path = job.get('tomogram') or job.get('tomogram_path') or job.get('volume_path')
            star_files = sorted(job_json.parent.glob('*_particles.star'))
            entry = {
                'ts_name':   ts_name,
                'img_b64':   None,
                'n_total':   0,
                'n_shown':   0,
                'tomo_path': tomo_path or '',
                'metadata':  {},
            }
            score_files = sorted(job_json.parent.glob('*_scores.mrc'))
            score_path  = score_files[0] if score_files else None

            if star_files and tomo_path and Path(tomo_path).exists():
                try:
                    from aretomo3_preprocess.shared.volume_qc import (
                        central_slab_projection, projection_to_b64png,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        data = _starfile.read(str(star_files[0]))
                    df = data[next(iter(data))] if isinstance(data, dict) else data
                    result = slab_with_picks_b64(
                        tomo_path, df,
                        slab_angst=qc_thick,
                        particle_diameter=args.particle_diameter,
                    )
                    # Score map
                    score_b64 = None
                    if score_path and score_path.exists():
                        proj = central_slab_projection(score_path, qc_thick)
                        if proj:
                            score_b64 = projection_to_b64png(proj['img'], pct=(50, 99.9))

                    if result:
                        entry.update({
                            'img_b64':   result['img_b64'],
                            'score_b64': score_b64,
                            'n_total':   result['n_total'],
                            'n_shown':   result['n_shown'],
                            'metadata': {
                                'slab':   f'{result["slab_a"]:.0f} Å  ({result["vox"]:.2f} Å/px)',
                                'picked': f'{result["n_shown"]} / {result["n_total"]} in slab',
                            },
                        })
                except Exception as exc:
                    print(f'  WARNING: QC render failed: {exc}')
            qc_entries.append(entry)

    # Summary
    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')
    if args.imod and not args.dry_run and ok:
        print(f'IMOD models: {mod_dir}/')

    # QC report
    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else in_dir / 'pytom_extract_qc.html')
        make_picks_html(
            entries    = qc_entries,
            out_path   = html_path,
            title      = 'pytom-extract QC',
            command    = ' '.join(sys.argv),
            slab_angst = qc_thick,
        )

    if args.dry_run:
        return

    update_section(
        section='pytom_extract',
        values={
            'command':     ' '.join(sys.argv),
            'args':        args_to_dict(args),
            'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
            'n_processed': len(ok),
            'failed':      failed,
            'input_dir':   str(in_dir),
        },
        backup_dir=in_dir,
    )
