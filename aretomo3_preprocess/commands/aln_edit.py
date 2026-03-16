"""
aln-edit — edit AreTomo3 .aln files in bulk.

Currently supports applying a tilt angle offset to all .aln files in a
directory.  The offset is added to the TILT column (last column) and the
# AlphaOffset header is updated to track the cumulative change.  All other
alignment parameters (ROT, TX, TY, GMAG, etc.) are left unchanged.

Backup / restore behaviour
--------------------------
On first run a .bak copy of each file is created.  Subsequent runs always
apply the offset to the .bak original so the offset never compounds.  Use
--restore to revert all files to their .bak originals.

Usage
-----
    aretomo3-preprocess aln-edit --input run001/ --apply-offset 2.5 --dry-run
    aretomo3-preprocess aln-edit --input run001/ --apply-offset 2.5
    aretomo3-preprocess aln-edit --input run001/ --restore
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

_ALPHA_RE = re.compile(r'^(#\s*AlphaOffset\s*=\s*)([-\d.]+)(.*)')


def add_parser(subparsers):
    p = subparsers.add_parser(
        'aln-edit',
        help='Edit AreTomo3 .aln files (apply tilt offset, restore backups)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', '-i', required=True,
                   help='Directory containing ts-*.aln files')

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--apply-offset', type=float, metavar='DEGREES',
                      help='Tilt angle offset in degrees to add to the TILT '
                           'column and AlphaOffset header of every .aln file')
    mode.add_argument('--restore', action='store_true',
                      help='Restore original .aln files from .bak backups')

    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be done without modifying any files')
    p.set_defaults(func=run)
    return p


def _confirm(prompt: str) -> bool:
    while True:
        reply = input(f'{prompt} [yes/no]: ').strip().lower()
        if reply == 'yes':
            return True
        if reply == 'no':
            return False
        print("  Please type 'yes' or 'no'.")


def _apply_offset(aln_files: list, offset: float, dry_run: bool):
    prefix = '[DRY RUN] ' if dry_run else ''

    already_backed_up = [f for f in aln_files
                         if f.with_suffix(f.suffix + '.bak').exists()]
    if already_backed_up:
        print(f'Note: {len(already_backed_up)} file(s) already have a .bak backup.')
        print('      Offset applied to ORIGINAL .bak angles — no compounding.')
        print()

    if not dry_run:
        print('This will:')
        print(f'  1. Back up each ts-xxx.aln to ts-xxx.aln.bak  (skipped if .bak exists)')
        print(f'  2. Overwrite each ts-xxx.aln: TILT += {offset:+.4f}°, AlphaOffset updated')
        print(f'  {len(aln_files)} files in {aln_files[0].parent}')
        print()
        if not _confirm('Proceed?'):
            print('Aborted.')
            sys.exit(0)
        print()

    for aln_path in aln_files:
        bak_path = aln_path.with_suffix(aln_path.suffix + '.bak')

        if not bak_path.exists():
            print(f'{prefix}Backup : {aln_path.name} -> {bak_path.name}')
            if not dry_run:
                shutil.copy2(aln_path, bak_path)
        else:
            print(f'         Skip backup (exists): {bak_path.name}')

        # Always read from .bak to prevent compounding
        src = bak_path if bak_path.exists() else aln_path
        lines = src.read_text().splitlines(keepends=True)

        out_lines = []
        n_data = 0
        for line in lines:
            stripped = line.strip()

            # Update AlphaOffset header
            m = _ALPHA_RE.match(stripped)
            if m:
                new_alpha = float(m.group(2)) + offset
                out_lines.append(f'# AlphaOffset = {new_alpha:8.2f}\n')
                continue

            # Pass through other comment / blank lines unchanged
            if not stripped or stripped.startswith('#'):
                out_lines.append(line)
                continue

            # Data row: shift TILT (last column)
            parts = stripped.split()
            try:
                parts[-1] = f'{float(parts[-1]) + offset:.2f}'
                out_lines.append('  '.join(parts) + '\n')
                n_data += 1
            except (ValueError, IndexError):
                out_lines.append(line)

        print(f'{prefix}Offset : {aln_path.name}  ({n_data} tilts, {offset:+.4f}°)')
        if not dry_run:
            aln_path.write_text(''.join(out_lines))

    print()
    if dry_run:
        print('Dry run complete — no files modified.')
    else:
        print('Done. Originals backed up as .bak files.')


def _restore_backup(aln_files: list, dry_run: bool):
    prefix = '[DRY RUN] ' if dry_run else ''

    restorable = [(f, f.with_suffix(f.suffix + '.bak'))
                  for f in aln_files
                  if f.with_suffix(f.suffix + '.bak').exists()]
    missing_bak = [f for f in aln_files
                   if not f.with_suffix(f.suffix + '.bak').exists()]

    if not restorable:
        print('No .bak backup files found — nothing to restore.')
        sys.exit(0)

    if missing_bak:
        print(f'Note: {len(missing_bak)} file(s) have no .bak and will be skipped.')
        print()

    if not dry_run:
        print('This will:')
        print(f'  Overwrite {len(restorable)} ts-xxx.aln file(s) with their .bak originals')
        print(f'  The .bak files will be removed after a successful restore')
        print(f'  Directory: {aln_files[0].parent}')
        print()
        if not _confirm('Proceed?'):
            print('Aborted.')
            sys.exit(0)
        print()

    for aln_path, bak_path in restorable:
        print(f'{prefix}Restore: {bak_path.name} -> {aln_path.name}')
        if not dry_run:
            shutil.copy2(bak_path, aln_path)
            bak_path.unlink()

    print()
    if dry_run:
        print(f'Dry run complete — {len(restorable)} file(s) would be restored.')
    else:
        print(f'Done. {len(restorable)} file(s) restored; .bak files removed.')


def run(args):
    in_dir = Path(args.input)

    if not in_dir.is_dir():
        print(f'ERROR: {in_dir} is not a directory')
        sys.exit(1)

    aln_files = sorted(f for f in in_dir.glob('ts-*.aln')
                       if not f.name.endswith('.bak'))

    if not aln_files:
        print(f'No ts-*.aln files found in {in_dir}')
        sys.exit(1)

    print(f'Directory : {in_dir}')
    print(f'Files     : {len(aln_files)}')
    if args.apply_offset is not None:
        print(f'Offset    : {args.apply_offset:+.4f}°')
    else:
        print(f'Mode      : restore from backup')
    print(f'Dry run   : {args.dry_run}')
    print()

    if args.apply_offset is not None:
        _apply_offset(aln_files, args.apply_offset, args.dry_run)
    else:
        _restore_backup(aln_files, args.dry_run)
