#!/usr/bin/env python3
"""
relion5_particles_to_mod.py — convert RELION5 extracted particles to IMOD .mod files.

Reads a RELION5 particles STAR file and a tomograms STAR file, then generates
one IMOD .mod file per tilt series for visual coordinate verification in 3dmod.

Usage:
    python relion5_particles_to_mod.py \\
        --particles Extract/job003/particles.star \\
        --tomograms Tomograms/job001/tomograms.star \\
        --output mod_files/ \\
        [--sphere-diameter 200]

The reconstructed tomogram pixel size is derived from the tomograms STAR file
(rlnTomoTiltSeriesPixelSize * rlnTomoTomogramBinning), or read from the MRC
header if --from-mrc is given.
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC file header."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    nx, ny, nz = struct.unpack_from('<3i', hdr, 0)
    return nx, ny, nz


def _load_tomograms(tomo_star):
    """
    Parse a RELION5 tomograms.star file.

    Returns a dict keyed by ts_name (rlnTomoName):
        {
          'ts_angpix':   float,
          'binning':     float,
          'rec_angpix':  float,
          'nx':          int,   # reconstructed tomogram X size
          'ny':          int,
          'nz':          int,
          'mrc_path':    Path or None,
        }
    """
    try:
        import warnings
        import starfile
        warnings.filterwarnings('ignore', category=FutureWarning)
    except ImportError:
        print('ERROR: starfile is required.  pip install starfile')
        sys.exit(1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        data = starfile.read(str(tomo_star))

    # starfile returns dict of DataFrames; pick the tomogram table
    if isinstance(data, dict):
        # Usually keyed 'data_' or by block name
        key = next((k for k in data if 'tomo' in k.lower()), None) or next(iter(data))
        df = data[key]
    else:
        df = data

    required = {'rlnTomoName', 'rlnTomoTiltSeriesPixelSize', 'rlnTomoTomogramBinning'}
    missing = required - set(df.columns)
    if missing:
        print(f'ERROR: tomograms STAR missing columns: {missing}')
        sys.exit(1)

    tomos = {}
    for _, row in df.iterrows():
        name      = str(row['rlnTomoName']).strip()
        ts_angpix = float(row['rlnTomoTiltSeriesPixelSize'])
        binning   = float(row['rlnTomoTomogramBinning'])
        rec_angpix = ts_angpix * binning

        # Reconstructed tomogram dimensions from tilt-series dimensions / binning
        nx = ny = nz = None
        for col_rec, col_ts in [('rlnTomoSizeX', 'rlnTomoSizeX'),
                                 ('rlnTomoSizeY', 'rlnTomoSizeY'),
                                 ('rlnTomoSizeZ', 'rlnTomoSizeZ')]:
            pass  # computed below

        if all(c in df.columns for c in ('rlnTomoSizeX', 'rlnTomoSizeY', 'rlnTomoSizeZ')):
            # rlnTomoSizeX/Y/Z are in tilt-series pixels in RELION5
            nx = round(float(row['rlnTomoSizeX']) / binning)
            ny = round(float(row['rlnTomoSizeY']) / binning)
            nz = round(float(row['rlnTomoSizeZ']) / binning)

        mrc_path = None
        if 'rlnTomoReconstructedTomogram' in df.columns:
            p = Path(str(row['rlnTomoReconstructedTomogram']).strip())
            if p.exists():
                mrc_path = p

        tomos[name] = {
            'ts_angpix':  ts_angpix,
            'binning':    binning,
            'rec_angpix': rec_angpix,
            'nx': nx, 'ny': ny, 'nz': nz,
            'mrc_path': mrc_path,
        }
    return tomos


def _load_particles(particles_star):
    """
    Parse a RELION5 particles STAR file.

    Returns a pandas DataFrame with at minimum:
        rlnTomoName, rlnCenteredCoordinateXAngst/YAngst/ZAngst
    """
    try:
        import warnings
        import starfile
        warnings.filterwarnings('ignore', category=FutureWarning)
    except ImportError:
        print('ERROR: starfile is required.  pip install starfile')
        sys.exit(1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        data = starfile.read(str(particles_star))

    # Pick the particles table (largest block, or 'data_particles')
    if isinstance(data, dict):
        key = next((k for k in data if 'particle' in k.lower()), None)
        if key is None:
            # largest table
            key = max(data, key=lambda k: len(data[k]))
        df = data[key]
    else:
        df = data

    required = {
        'rlnTomoName',
        'rlnCenteredCoordinateXAngst',
        'rlnCenteredCoordinateYAngst',
        'rlnCenteredCoordinateZAngst',
    }
    missing = required - set(df.columns)
    if missing:
        print(f'ERROR: particles STAR missing columns: {missing}')
        sys.exit(1)

    return df


def _write_mod(ts_name, rows_x, rows_y, rows_z, nx, ny, nz,
               rec_angpix, out_dir, sphere_diameter):
    """Convert centred Angstrom coordinates to voxel coords and write .mod."""
    point2model = shutil.which('point2model')
    if not point2model:
        print('ERROR: point2model not found on PATH — load IMOD first')
        sys.exit(1)

    # Centred Angstrom → reconstructed tomogram voxel
    xs = rows_x / rec_angpix + nx / 2
    ys = rows_y / rec_angpix + ny / 2
    zs = rows_z / rec_angpix + nz / 2

    radius_px = (sphere_diameter / 2.0) / rec_angpix if sphere_diameter else None

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f'{ts_name}_particles.txt'
    mod_path = out_dir / f'{ts_name}_particles.mod'

    with open(txt_path, 'w') as fh:
        for x, y, z in zip(xs, ys, zs):
            if radius_px is not None:
                fh.write(f'{x:.4f} {y:.4f} {z:.4f} {radius_px:.2f}\n')
            else:
                fh.write(f'{x:.4f} {y:.4f} {z:.4f}\n')

    cmd = [point2model, '-scat']
    if radius_px is not None:
        cmd += ['-sizes']
    cmd += [str(txt_path), str(mod_path)]
    ret = subprocess.run(cmd, capture_output=True)
    txt_path.unlink(missing_ok=True)

    if ret.returncode != 0:
        print(f'  WARNING: point2model failed for {ts_name}: '
              f'{ret.stderr.decode().strip()}')
        return False

    print(f'  {mod_path}  ({len(rows_x)} particles,  rec_angpix={rec_angpix:.3f} Å,'
          f'  tomo {nx}×{ny}×{nz})')
    return True


def main():
    ap = argparse.ArgumentParser(
        description='Convert RELION5 extracted particles to per-TS IMOD .mod files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--particles', '-p', required=True, metavar='STAR',
                    help='RELION5 particles STAR file (e.g. Extract/job003/particles.star)')
    ap.add_argument('--tomograms', '-t', required=True, metavar='STAR',
                    help='RELION5 tomograms STAR file (e.g. Tomograms/job001/tomograms.star)')
    ap.add_argument('--output', '-o', default='mod_files', metavar='DIR',
                    help='Output directory for .mod files')
    ap.add_argument('--sphere-diameter', type=float, default=None, metavar='ANGST',
                    help='Sphere diameter in Angstroms (shown in 3dmod as spheres). '
                         'Leave unset for plain points.')
    ap.add_argument('--from-mrc', action='store_true',
                    help='Read reconstructed tomogram dimensions from the MRC file '
                         'header (requires rlnTomoReconstructedTomogram in tomograms STAR). '
                         'Default: derive from rlnTomoSizeX/Y/Z / rlnTomoTomogramBinning.')
    ap.add_argument('--include', nargs='+', default=None, metavar='TS',
                    help='Process only these TS names (e.g. --include ts-004 ts-027). '
                         'Default: all TS present in particles STAR.')
    ap.add_argument('--exclude', nargs='+', default=None, metavar='TS',
                    help='Skip these TS names (e.g. --exclude ts-010 ts-011). '
                         'Applied after --include.')
    args = ap.parse_args()

    particles_star = Path(args.particles)
    tomograms_star = Path(args.tomograms)
    out_dir        = Path(args.output)

    if not particles_star.exists():
        print(f'ERROR: particles STAR not found: {particles_star}')
        sys.exit(1)
    if not tomograms_star.exists():
        print(f'ERROR: tomograms STAR not found: {tomograms_star}')
        sys.exit(1)

    print(f'Reading particles: {particles_star}')
    df = _load_particles(particles_star)
    print(f'  {len(df)} particles total')

    print(f'Reading tomograms: {tomograms_star}')
    tomos = _load_tomograms(tomograms_star)
    print(f'  {len(tomos)} tomograms in set')

    ts_names = sorted(df['rlnTomoName'].unique())
    if args.include:
        ts_names = [t for t in ts_names if t in set(args.include)]
        if not ts_names:
            print('ERROR: none of the specified --include names found in particles STAR')
            sys.exit(1)
    if args.exclude:
        excluded = set(args.exclude)
        ts_names = [t for t in ts_names if t not in excluded]
        if not ts_names:
            print('ERROR: no TS remaining after --exclude filter')
            sys.exit(1)

    print(f'\nWriting .mod files to {out_dir}/')
    n_ok = 0
    for ts_name in ts_names:
        sub = df[df['rlnTomoName'] == ts_name]
        if ts_name not in tomos:
            print(f'  WARNING: {ts_name} not in tomograms STAR — skipping')
            continue

        info = tomos[ts_name]
        rec_angpix = info['rec_angpix']

        # Tomogram dimensions
        if args.from_mrc and info['mrc_path']:
            nx, ny, nz = _read_mrc_dims(info['mrc_path'])
        elif info['nx'] is not None:
            nx, ny, nz = info['nx'], info['ny'], info['nz']
        elif info['mrc_path']:
            nx, ny, nz = _read_mrc_dims(info['mrc_path'])
        else:
            print(f'  WARNING: cannot determine dimensions for {ts_name} — '
                  f'no rlnTomoSizeX/Y/Z and no rlnTomoReconstructedTomogram found')
            continue

        ok = _write_mod(
            ts_name=ts_name,
            rows_x=sub['rlnCenteredCoordinateXAngst'].values,
            rows_y=sub['rlnCenteredCoordinateYAngst'].values,
            rows_z=sub['rlnCenteredCoordinateZAngst'].values,
            nx=nx, ny=ny, nz=nz,
            rec_angpix=rec_angpix,
            out_dir=out_dir,
            sphere_diameter=args.sphere_diameter,
        )
        if ok:
            n_ok += 1

    print(f'\nDone: {n_ok}/{len(ts_names)} .mod files written to {out_dir}/')
    if n_ok:
        print('\nTo open a tomogram with its picks in 3dmod:')
        print(f'  3dmod <tomogram.mrc> {out_dir}/<ts-name>_particles.mod')


if __name__ == '__main__':
    main()
