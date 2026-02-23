"""
rename-ts — create ts-XXXX.mdoc symlinks for Position_*.mdoc files.

Symlinks are created inside the input directory alongside the originals
so AreTomo3 can find them.  A lookup table mapping ts-name -> original
absolute path is saved to aretomo3_project.json under the 'rename_ts' key.

For multi-session datasets, use --start to offset numbering and avoid
collisions (e.g. session 2: --start 100).
"""

import sys
import datetime
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict


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

    print(f'{prefix}Creating {n} symlinks in {in_dir}/')
    print()

    lookup = {}  # ts-name -> original absolute path (str)
    for i, mdoc_path in enumerate(mdoc_files, start=args.start):
        ts_name = f'ts-{i:0{digits}d}.mdoc'
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
        update_section(
            section='rename_ts',
            values={
                'command':    ' '.join(sys.argv),
                'args':       args_to_dict(args),
                'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
                'n_symlinks': n,
                'lookup':     lookup,
            },
        )
