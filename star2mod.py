#!/usr/bin/env python3
"""
star2mod.py — convert pytom-match RELION5 STAR files to IMOD .mod point models.

Finds all *_particles.star files in a pytom-match output directory and converts
each to an IMOD .mod file using point2model.  Requires IMOD on PATH and the
starfile Python package (available in the pytom_tm conda environment).

Adapted from rln2mod by Phaips (https://github.com/Phaips/rln2mod).

Usage
-----
  python star2mod.py pytom_match_test/
  python star2mod.py pytom_match_test/ --output-dir my_mods/
  python star2mod.py pytom_match_test/ --include ts-003 ts-004
"""

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path


def _mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC header without mrcfile dependency."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)


def star_to_mod(star_path, job_json_path, mod_dir, keep_txt=False, sphere_diameter=None):
    """Convert a RELION5 particles STAR file to an IMOD .mod point model."""
    try:
        import starfile
        warnings.filterwarnings('ignore', category=FutureWarning)
    except ImportError:
        print('ERROR: starfile not installed.')
        print('  conda run -n pytom_tm pip install starfile')
        print('  or: pip install starfile')
        return False

    # Tomogram dimensions from job JSON
    with open(job_json_path) as fh:
        job = json.load(fh)
    tomo_path = job.get('tomogram') or job.get('tomogram_path') or job.get('volume_path')
    if not tomo_path or not Path(tomo_path).exists():
        print(f'  WARNING: cannot find tomogram path in {job_json_path.name} — skipping')
        return False

    nx, ny, nz = _mrc_dims(tomo_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        data = starfile.read(str(star_path))
    df = data[next(iter(data))] if isinstance(data, dict) else data

    required = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
                'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    missing = required - set(df.columns)
    if missing:
        print(f'  WARNING: STAR file missing columns {missing}')
        print('           Re-run pytom-extract with --relion5-compat')
        return False

    px = df['rlnTomoTiltSeriesPixelSize']
    xs = df['rlnCenteredCoordinateXAngst'] / px + nx / 2
    ys = df['rlnCenteredCoordinateYAngst'] / px + ny / 2
    zs = df['rlnCenteredCoordinateZAngst'] / px + nz / 2

    radius_px = (sphere_diameter / 2.0) / float(px.iloc[0]) if sphere_diameter else None

    mod_dir.mkdir(parents=True, exist_ok=True)
    stem = star_path.stem
    txt  = mod_dir / f'{stem}.txt'
    mod  = mod_dir / f'{stem}.mod'

    with open(txt, 'w') as fh:
        for x, y, z in zip(xs, ys, zs):
            if radius_px is not None:
                fh.write(f'{x:.6f} {y:.6f} {z:.6f} {radius_px:.2f}\n')
            else:
                fh.write(f'{x:.6f} {y:.6f} {z:.6f}\n')

    point2model = shutil.which('point2model')
    if not point2model:
        print('  WARNING: point2model not found on PATH — load IMOD first')
        txt.unlink(missing_ok=True)
        return False

    cmd = [point2model, '-scat']
    if radius_px is not None:
        cmd += ['-sizes']
    cmd += [str(txt), str(mod)]
    ret = subprocess.run(cmd, capture_output=True)
    if not keep_txt:
        txt.unlink(missing_ok=True)
    if ret.returncode != 0:
        out = ret.stdout.decode().strip()
        err = ret.stderr.decode().strip()
        print(f'  WARNING: point2model failed (code {ret.returncode})')
        if out:
            print(f'    stdout: {out}')
        if err:
            print(f'    stderr: {err}')
        if not out and not err:
            print(f'    command: {" ".join(cmd)}')
        return False

    print(f'  → {mod}  ({len(df)} particles)')
    if keep_txt:
        print(f'  → {txt}  (coordinates)')
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('input', help='pytom-match output directory')
    ap.add_argument('--output-dir', '-o', default=None,
                    help='Directory for .mod files (default: <input>/mod/)')
    ap.add_argument('--include', nargs='+',
                    help='Process only these TS names (wildcards supported)')
    ap.add_argument('--exclude', nargs='+',
                    help='Exclude these TS names (wildcards supported)')
    ap.add_argument('--keep-txt', action='store_true',
                    help='Keep the intermediate coordinate .txt file alongside the .mod')
    ap.add_argument('--sphere-diameter', type=float, default=280.0, metavar='ANGST',
                    help='Sphere display diameter in Å (default: 280); '
                         'set to 0 to skip sphere sizing')
    args = ap.parse_args()

    in_dir  = Path(args.input).resolve()
    mod_dir = Path(args.output_dir).resolve() if args.output_dir else in_dir / 'mod'

    if not in_dir.is_dir():
        print(f'ERROR: {in_dir} not found')
        sys.exit(1)

    # Collect (ts_name, star_path, job_json) tuples
    entries = []
    for ts_dir in sorted(in_dir.iterdir()):
        if not ts_dir.is_dir():
            continue
        stars = sorted(ts_dir.glob('*_particles.star'))
        jobs  = sorted(ts_dir.glob('*_job.json'))
        if stars and jobs:
            for star in stars:
                entries.append((ts_dir.name, star, jobs[0]))

    if not entries:
        print(f'ERROR: no *_particles.star files found in {in_dir}/')
        print('       Run pytom-extract with --relion5-compat first.')
        sys.exit(1)

    # include / exclude filtering
    if args.include:
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        entries = [(n, s, j) for n, s, j in entries
                   if any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in inc)]
    if args.exclude:
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        entries = [(n, s, j) for n, s, j in entries
                   if not any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in exc)]

    if not entries:
        print('ERROR: no TS remaining after include/exclude filtering')
        sys.exit(1)

    print(f'Converting {len(entries)} STAR file(s) → {mod_dir}/')
    print()

    ok, failed = [], []
    for ts_name, star, job_json in entries:
        print(f'{ts_name}')
        sphere_d = args.sphere_diameter if args.sphere_diameter else None
        if star_to_mod(star, job_json, mod_dir, keep_txt=args.keep_txt,
                       sphere_diameter=sphere_d):
            ok.append(ts_name)
        else:
            failed.append(ts_name)

    print()
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')


if __name__ == '__main__':
    main()
