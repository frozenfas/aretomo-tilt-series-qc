"""
topaz-denoise3d — denoise reconstructed volumes using Topaz denoise3d.

Applies a pretrained 3D denoising model to ts-xxx_Vol.mrc files using
Topaz (https://topaz-em.readthedocs.io).  Only pretrained models are
supported; no training is performed.

Pretrained models
-----------------
  unet-3d      — general-purpose (default)
  unet-3d-10a  — trained on 10 Å/px data
  unet-3d-20a  — trained on 20 Å/px data

Example
-------
  aretomo3-preprocess topaz-denoise3d \\
      --input run001 \\
      --output run001/topaz_denoise \\
      --model unet-3d \\
      --gpu 0 \\
      --select-ts run001_analysis/ts-select.csv \\
      --dry-run
"""

import sys
import subprocess
import shutil
import csv
from pathlib import Path
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_selected(csv_path: Path) -> set:
    selected = set()
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            ts = row.get('ts_name', '').strip()
            if ts and row.get('selected', '').strip() == '1':
                selected.add(ts)
    return selected


def _ts_name_from_vol(vol_path: Path, vol_suffix: str) -> str:
    stem = vol_path.stem
    for tag in (
        f'_EVN{vol_suffix}', f'_ODD{vol_suffix}',
        f'{vol_suffix}_EVN', f'{vol_suffix}_ODD',
        vol_suffix,
    ):
        if tag and stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'topaz-denoise3d',
        help='Denoise reconstructed volumes with Topaz pretrained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', '-i', required=True,
                   help='Directory containing ts-xxx_Vol.mrc files')
    p.add_argument('--output', '-o', default=None,
                   help='Output directory (default: <input>/topaz_denoise)')
    p.add_argument('--model', '-m',
                   choices=['unet-3d', 'unet-3d-10a', 'unet-3d-20a'],
                   default='unet-3d',
                   help='Pretrained denoising model (default: unet-3d)')
    p.add_argument('--gpu', '-d', type=int, default=0,
                   help='GPU device index (default: 0; use -1 for CPU)')
    p.add_argument('--select-ts', default=None, metavar='CSV',
                   help='ts-select.csv; only selected TS are denoised')
    p.add_argument('--vol-suffix', default='_Vol',
                   help='Volume filename suffix (default: _Vol → ts-xxx_Vol.mrc)')
    p.add_argument('--patch-size', type=int, default=96,
                   help='Patch size for denoising (default: 96)')
    p.add_argument('--patch-padding', type=int, default=48,
                   help='Patch padding to reduce edge artefacts (default: 48)')
    p.add_argument('--topaz', dest='topaz_bin',
                   default='/opt/miniconda3/envs/topaz/bin/topaz',
                   help='Path to topaz executable')
    p.add_argument('--dry-run', action='store_true',
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

    out_dir = Path(args.output) if args.output else in_dir / 'topaz_denoise'
    sfx     = args.vol_suffix

    # ── TS selection ──────────────────────────────────────────────────────────
    selected = None
    if args.select_ts:
        csv_path = Path(args.select_ts)
        if not csv_path.exists():
            print(f'ERROR: --select-ts {csv_path} not found')
            sys.exit(1)
        selected = _load_selected(csv_path)
        print(f'TS selection : {len(selected)} selected from {csv_path.name}')

    # ── Find volumes ──────────────────────────────────────────────────────────
    vol_files = sorted(in_dir.glob(f'ts-*{sfx}.mrc'))

    # Exclude EVN/ODD half-sets
    vol_files = [
        f for f in vol_files
        if not (f.stem.endswith(f'_EVN{sfx}') or f.stem.endswith(f'_ODD{sfx}'))
    ]

    if not vol_files:
        print(f'ERROR: no volumes found in {in_dir}/ matching ts-*{sfx}.mrc')
        sys.exit(1)

    if selected is not None:
        before    = len(vol_files)
        vol_files = [f for f in vol_files
                     if _ts_name_from_vol(f, sfx) in selected]
        print(f'After selection: {len(vol_files)} / {before} volumes')

    if not vol_files:
        print('No volumes to process after selection filter.')
        sys.exit(0)

    print(f'Volumes to process : {len(vol_files)}')
    print(f'Model              : {args.model}')
    print(f'Device             : {args.gpu}')
    print(f'Output directory   : {out_dir}/')
    print()

    # ── Check binary ──────────────────────────────────────────────────────────
    if not args.dry_run:
        if (shutil.which(args.topaz_bin) is None
                and not Path(args.topaz_bin).is_file()):
            print(f'ERROR: topaz binary not found: {args.topaz_bin!r}')
            print('       Install topaz or use --topaz /path/to/topaz')
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────────────
    prefix = '[DRY RUN] ' if args.dry_run else ''
    n_ok = n_fail = 0

    for vol_path in vol_files:
        cmd = [
            args.topaz_bin, 'denoise3d',
            str(vol_path),
            '-o',  str(out_dir),
            '-m',  args.model,
            '-d',  str(args.gpu),
            '-s',  str(args.patch_size),
            '-p',  str(args.patch_padding),
        ]

        print(f'{prefix}{vol_path.name}')
        if args.dry_run:
            print(f'      {" ".join(cmd)}')
            n_ok += 1
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  FAIL  {vol_path.name}  (exit {result.returncode})')
            for line in (result.stderr or '').strip().splitlines()[:5]:
                print(f'        {line}')
            n_fail += 1
        else:
            n_ok += 1

    print()
    print(f'Done: {n_ok} processed, {n_fail} failed')
    if n_fail:
        sys.exit(1)
