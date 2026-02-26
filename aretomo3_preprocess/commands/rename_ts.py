"""
rename-ts — create ts-XXXX.mdoc symlinks for Position_*.mdoc files.

Symlinks are created inside the input directory alongside the originals
so AreTomo3 can find them.

Each invocation of rename-ts represents one data-collection session (grid).
Running it multiple times with increasing --start values assigns consecutive
ts-XXXX numbers across sessions.  Each run is recorded as a separate grid in
aretomo3_project.json, so that downstream commands (e.g. analyse) know which
tilt series belong to the same grid and can cluster lamellae correctly.

project.json structure after rename-ts:
    rename_ts.grids.<N>  — per-grid metadata + per-grid lookup table
    rename_ts.ts_to_grid — flat map ts-stem -> grid number (for clustering)
    rename_ts.lookup     — merged ts-name.mdoc -> original absolute path
"""

import csv
import sys
import datetime
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import (
    load as _load_project,
    update_section, args_to_dict,
)


def _write_csv(lookup: dict, ts_to_grid: dict, path: Path):
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ts_name', 'grid', 'original_path'])
        for ts_name, orig_path in lookup.items():
            stem = ts_name.replace('.mdoc', '')
            grid = ts_to_grid.get(stem, '')
            writer.writerow([ts_name, grid, orig_path])
    print(f'Lookup CSV written : {path}')


def add_parser(subparsers):
    p = subparsers.add_parser(
        'rename-ts',
        help='Create ts-XXXX.mdoc symlinks from Position_*.mdoc files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', '-i', default='frames',
                   help='Directory containing *.mdoc files (default: frames)')
    p.add_argument('--start', type=int, default=1,
                   help='Starting number for ts-XXXX naming (default: 1)')
    p.add_argument('--digits', type=int, default=None,
                   help='Zero-pad width (auto-detected if omitted)')
    p.add_argument('--dry-run', action='store_true',
                   help='Preview without creating symlinks')
    p.set_defaults(func=run)
    return p


def run(args):
    in_dir = Path(args.input)
    if not in_dir.is_dir():
        print(f'Error: {in_dir} is not a directory')
        sys.exit(1)

    # Glob all .mdoc files; skip existing symlinks (e.g. ts-*.mdoc from a previous run)
    mdoc_files = sorted(p for p in in_dir.glob('*.mdoc') if not p.is_symlink())

    if not mdoc_files:
        print(f'No .mdoc files found in {in_dir}')
        sys.exit(1)

    n = len(mdoc_files)
    digits = args.digits or len(str(args.start + n - 1))
    prefix = '[DRY RUN] ' if args.dry_run else ''

    # Determine grid number from existing project state
    proj          = _load_project()
    existing      = proj.get('rename_ts', {})
    existing_grids = existing.get('grids', {})
    grid_no       = max((int(g) for g in existing_grids), default=0) + 1

    print(f'{prefix}Grid {grid_no} — creating {n} symlinks in {in_dir}/')
    print()

    lookup = {}   # ts-name.mdoc -> original absolute path (str), this grid only
    for i, mdoc_path in enumerate(mdoc_files, start=args.start):
        ts_name  = f'ts-{i:0{digits}d}.mdoc'
        symlink_path = in_dir / ts_name
        resolved = mdoc_path.resolve()

        if not args.dry_run:
            if symlink_path.is_symlink() or symlink_path.exists():
                raise FileExistsError(f'Symlink already exists: {symlink_path}')
            symlink_path.symlink_to(resolved)

        lookup[ts_name] = str(resolved)
        print(f'  {prefix}{ts_name} -> {resolved.name}')

    print()
    print(f'{prefix}{n} symlinks {"would be " if args.dry_run else ""}created.')

    if not args.dry_run:
        # Merge with existing grids
        merged_lookup     = {**existing.get('lookup', {}), **lookup}
        merged_ts_to_grid = dict(existing.get('ts_to_grid', {}))
        for ts_name in lookup:
            merged_ts_to_grid[ts_name.replace('.mdoc', '')] = grid_no

        new_grids = {
            **existing_grids,
            str(grid_no): {
                'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
                'command':    ' '.join(sys.argv),
                'args':       args_to_dict(args),
                'input_dir':  str(in_dir.resolve()),
                'start':      args.start,
                'digits':     digits,
                'n_symlinks': n,
                'lookup':     lookup,
            },
        }

        update_section(
            section='rename_ts',
            values={
                'grids':       new_grids,
                'ts_to_grid':  merged_ts_to_grid,
                'lookup':      merged_lookup,
            },
        )

        csv_path = Path.cwd() / 'ts_rename_lookup.csv'
        _write_csv(merged_lookup, merged_ts_to_grid, csv_path)
        _write_csv(merged_lookup, merged_ts_to_grid, in_dir / 'ts_rename_lookup.csv')

        print(f'\nGrid {grid_no} recorded in project.json'
              f'  ({len(existing_grids)} previous grid(s) + this one)')
