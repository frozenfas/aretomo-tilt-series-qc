"""
imod-mtffilter — apply IMOD's mtffilter Wiener deconvolution to reconstructed volumes.

Applies mtffilter (available since IMOD 4.12.36) to ts-xxx_Vol.mrc files in an
input directory.  The filter sharpens tomograms using a CTF-based Wiener filter
derived from the Warp deconvolution approach (Tegunov & Cramer, 2019).

Defocus values are read (in order of preference) from:
  1. ts-select.csv  (ref_defocus_um column) via --select-ts
  2. project.json   (defocus_data section)  via: enrich --defocus-data
  3. --defocus       global fallback value

Pixel size is read from the MRC header of each volume (correct for binning).
Falls back to --apix if mrcfile is not installed.

Example
-------
  aretomo3-preprocess imod-mtffilter \\
      --input run001 \\
      --select-ts run001_analysis/ts-select.csv \\
      --deconv 1.0 --snr 1.0 \\
      --dry-run
"""

import csv
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_state import get_defocus_data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tsselect(csv_path: Path):
    """
    Load ts-select.csv.

    Returns (defocus_map, selected_set) where:
      defocus_map  — {ts_name: float µm}   (entries with non-empty ref_defocus_um)
      selected_set — set of ts_name where selected == 1
    """
    defocus_map  = {}
    selected_set = set()
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            ts = row.get('ts_name', '').strip()
            if not ts:
                continue
            if row.get('selected', '').strip() == '1':
                selected_set.add(ts)
            val = row.get('ref_defocus_um', '').strip()
            if val:
                try:
                    defocus_map[ts] = float(val)
                except ValueError:
                    pass
    return defocus_map, selected_set


def _read_one_voxel_size(mrc_path: Path):
    """Read pixel size (Å) from a single MRC header; return None on failure."""
    try:
        import mrcfile
        with mrcfile.open(mrc_path, mode='r', permissive=True) as m:
            ps = float(m.voxel_size.x)
            if ps > 0:
                return ps
    except Exception:
        pass
    return None


def _ts_name_from_vol(vol_path: Path, vol_suffix: str) -> str:
    """
    Extract ts_name from a volume Path.

    Handles both AreTomo3 naming conventions:
      ts-001_Vol.mrc      (single-bin main)  → ts-001
      ts-001_EVN_Vol.mrc  (single-bin EVN)   → ts-001
      ts-001_ODD_Vol.mrc  (single-bin ODD)   → ts-001
      ts-001_b4.mrc       (multi-bin main)   → ts-001
      ts-001_b4_EVN.mrc   (multi-bin EVN)    → ts-001
      ts-001_b4_ODD.mrc   (multi-bin ODD)    → ts-001
    """
    stem = vol_path.stem
    for tag in (
        f'_EVN{vol_suffix}',   # e.g. _EVN_Vol  (single-bin EVN)
        f'_ODD{vol_suffix}',   # e.g. _ODD_Vol  (single-bin ODD)
        f'{vol_suffix}_EVN',   # e.g. _b4_EVN   (multi-bin EVN)
        f'{vol_suffix}_ODD',   # e.g. _b4_ODD   (multi-bin ODD)
        vol_suffix,            # e.g. _Vol, _b4 (main volume)
    ):
        if tag and stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'imod-mtffilter',
        help='Apply IMOD mtffilter Wiener deconvolution to reconstructed volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', '-i', required=True,
                   help='Directory containing ts-xxx_Vol.mrc files')
    p.add_argument('--output', '-o', default=None,
                   help='Output directory (default: <input>/mtffilter_vol)')
    p.add_argument('--select-ts', default=None, metavar='CSV',
                   help='ts-select.csv from select-ts; only selected TS are '
                        'processed, and ref_defocus_um is read from this file')

    filt = p.add_argument_group('filter parameters')
    filt.add_argument('--deconv', type=float, default=1.0,
                      help='Deconvolution strength factor (-deconv, default 1.0)')
    filt.add_argument('--snr', type=float, default=1.0,
                      help='Signal-to-noise ratio (-snr, default 1.0)')
    filt.add_argument('--defocus', type=float, default=None, metavar='UM',
                      help='Global defocus fallback in µm; used when no per-TS '
                           'value is available from ts-select.csv or project.json')
    filt.add_argument('--apix', type=float, default=None,
                      help='Pixel size in Å; auto-read from MRC header if omitted')

    vol = p.add_argument_group('volume selection')
    vol.add_argument('--vol-suffix', default='_Vol',
                     help='Volume filename suffix; globs ts-xxx{suffix}.mrc '
                          '(default: _Vol  →  ts-xxx_Vol.mrc)')
    vol.add_argument('--halves', action='store_true',
                     help='Also process EVN/ODD half-set volumes '
                          '(ts-xxx_EVN_Vol.mrc and ts-xxx_ODD_Vol.mrc)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--mtffilter', dest='mtffilter_bin', default='mtffilter',
                     help='Path to or name of the mtffilter executable '
                          '(default: mtffilter)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without executing')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input)
    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} is not a directory')
        sys.exit(1)

    out_dir = Path(args.output) if args.output else in_dir / 'mtffilter_vol'

    # ── Defocus and TS selection ──────────────────────────────────────────────
    defocus_map  = {}    # {ts_name: float µm}
    selected_set = None  # None = process all

    if args.select_ts:
        csv_path = Path(args.select_ts)
        if not csv_path.exists():
            print(f'ERROR: --select-ts {csv_path} not found')
            sys.exit(1)
        defocus_map, selected_set = _load_tsselect(csv_path)
        print(f'TS selection    : {len(selected_set)} selected from {csv_path.name}')
        n_def = len(defocus_map)
        print(f'Defocus values  : {n_def} from ts-select.csv')

    # project.json defocus_data as fallback
    proj_defocus = get_defocus_data() or {}
    if proj_defocus and not defocus_map:
        print(f'Defocus values  : {len(proj_defocus)} from project.json '
              f'(defocus_data)')

    # ── Find volume files ─────────────────────────────────────────────────────
    sfx = args.vol_suffix
    glob_patterns = [f'ts-*{sfx}.mrc']
    if args.halves:
        if sfx == '_Vol':
            # Single-bin: EVN/ODD appear before Vol in the name
            glob_patterns += ['ts-*_EVN_Vol.mrc', 'ts-*_ODD_Vol.mrc']
        else:
            # Multi-bin: EVN/ODD appear after the bin suffix
            glob_patterns += [f'ts-*{sfx}_EVN.mrc', f'ts-*{sfx}_ODD.mrc']

    vol_files = []
    for pat in glob_patterns:
        vol_files.extend(in_dir.glob(pat))
    vol_files = sorted(set(vol_files))

    # The wildcard glob ts-*_Vol.mrc also matches _EVN_Vol.mrc / _ODD_Vol.mrc.
    # Exclude half-sets unless --halves was requested.
    if not args.halves:
        vol_files = [
            f for f in vol_files
            if not (f.stem.endswith(f'_EVN{sfx}') or f.stem.endswith(f'_ODD{sfx}'))
        ]

    if not vol_files:
        print(f'ERROR: no volumes found in {in_dir}/ matching '
              f'{", ".join(glob_patterns)}')
        sys.exit(1)

    # ── Filter by TS selection ────────────────────────────────────────────────
    if selected_set is not None:
        before    = len(vol_files)
        vol_files = [f for f in vol_files
                     if _ts_name_from_vol(f, sfx) in selected_set]
        print(f'After selection : {len(vol_files)} / {before} volumes')

    if not vol_files:
        print('No volumes to process after selection filter.')
        sys.exit(0)

    print(f'Volumes to process: {len(vol_files)}')
    print(f'Output directory  : {out_dir}/')

    # ── Resolve pixel size ────────────────────────────────────────────────────
    # Read the header of the first volume only — all volumes from the same run
    # have the same binning and therefore the same pixel size.
    header_apix = _read_one_voxel_size(vol_files[0])
    if header_apix is not None:
        if args.apix is not None and abs(args.apix - header_apix) > 0.01:
            print(f'WARNING: --apix {args.apix} Å differs from MRC header '
                  f'{header_apix} Å ({vol_files[0].name}); using MRC header value.')
        apix = header_apix
        print(f'Pixel size        : {apix} Å  (from MRC header {vol_files[0].name})')
    elif args.apix is not None:
        apix = args.apix
        print(f'Pixel size        : {apix} Å  (from --apix)')
    else:
        print('ERROR: cannot determine pixel size — install mrcfile or supply --apix')
        sys.exit(1)
    print()

    # ── Check binary ──────────────────────────────────────────────────────────
    if not args.dry_run:
        if (shutil.which(args.mtffilter_bin) is None
                and not Path(args.mtffilter_bin).is_file()):
            print(f'ERROR: mtffilter binary not found: {args.mtffilter_bin!r}')
            print('       Ensure IMOD is on PATH or use '
                  '--mtffilter /path/to/mtffilter')
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each volume ───────────────────────────────────────────────────
    prefix = '[DRY RUN] ' if args.dry_run else ''
    n_ok = n_skip = n_fail = 0

    for vol_path in vol_files:
        ts_name  = _ts_name_from_vol(vol_path, sfx)
        out_path = out_dir / vol_path.name

        # Defocus: ts-select > project.json > global --defocus
        defocus = (defocus_map.get(ts_name)
                   or proj_defocus.get(ts_name)
                   or args.defocus)
        if defocus is None:
            print(f'  SKIP  {vol_path.name}: no defocus value '
                  f'(add via --select-ts, --defocus, or: '
                  f'enrich --defocus-data <run_dir>)')
            n_skip += 1
            continue

        cmd = [
            args.mtffilter_bin,
            str(vol_path),
            str(out_path),
            '-deconv',  str(args.deconv),
            '-snr',     str(args.snr),
            '-defocus', str(round(defocus, 4)),
            '-pixel',   str(round(apix,   4)),
        ]

        print(f'{prefix}{vol_path.name}  '
              f'defocus={defocus:.3f}µm  pixel={apix:.4f}Å')
        if args.dry_run:
            print(f'      {" ".join(cmd)}')
            n_ok += 1
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  FAIL  {vol_path.name}  (exit {result.returncode})')
            stderr_lines = (result.stderr or '').strip().splitlines()
            for line in stderr_lines[:5]:
                print(f'        {line}')
            n_fail += 1
        else:
            n_ok += 1

    print()
    print(f'Done: {n_ok} processed, {n_skip} skipped, {n_fail} failed')
    if n_fail:
        sys.exit(1)
