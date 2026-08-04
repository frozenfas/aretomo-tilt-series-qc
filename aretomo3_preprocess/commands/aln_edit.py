"""
aln-edit — edit AreTomo3 .aln files (and their IMOD companions) in bulk.

Currently supports applying a tilt angle offset to all .aln files in a
directory. Both the # AlphaOffset header AND the TILT column (last column
of each data row) are updated, and the corresponding IMOD ts-XXX_Imod/
ts-XXX_st.tlt file (if present) is updated by the same amount -- this
matches AreTomo3's own -TiltCor behaviour, confirmed directly (dataset-wide,
comparing the same TS processed with -TiltCor 0 vs 1): when TiltCor is on,
AreTomo3 bakes the correction into both the .aln TILT column and the IMOD
_st.tlt file it derives from it, not just the header. See CLAUDE.md's
alpha_offset convention section for the full test and rationale. Consumers
should therefore always use the TILT column / IMOD _st.tlt value directly
and never add AlphaOffset again -- this tool keeps that true after a manual
correction too, rather than requiring downstream code to special-case
aln-edited files.

This codebase prefers IMOD-format files over AreTomo3-native ones where
both exist (IMOD's formats are more stable/consistently documented across
tool versions) -- see CLAUDE.md. That's also why this tool updates
ts-XXX_st.tlt, not just the .aln: consumers reading IMOD's own .tlt file
would otherwise silently miss a manual aln-edit correction entirely.

Backup / restore behaviour
--------------------------
On first run a .bak copy of each file is created (both ts-xxx.aln and, if
present, ts-xxx_Imod/ts-xxx_st.tlt). Subsequent runs always apply the
offset to the .bak originals so the offset never compounds. Use --restore
to revert all files to their .bak originals.

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
        help='Edit AreTomo3 .aln files + IMOD .tlt companions (apply tilt offset, restore backups)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', '-i', required=True,
                   help='Directory containing ts-*.aln files (and ts-*_Imod/ subdirs)')

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--apply-offset', type=float, metavar='DEGREES',
                      help='Tilt angle offset in degrees to add to the AlphaOffset '
                           'header AND the TILT column of every .aln file, and to '
                           'the matching IMOD ts-XXX_Imod/ts-XXX_st.tlt file if '
                           'present (matches AreTomo3\'s own -TiltCor behaviour).')
    mode.add_argument('--restore', action='store_true',
                      help='Restore original .aln/_st.tlt files from .bak backups')

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


def _imod_tlt_path(aln_path: Path) -> Path:
    ts_name = aln_path.stem  # ts-XXX.aln -> ts-XXX
    return aln_path.parent / f'{ts_name}_Imod' / f'{ts_name}_st.tlt'


def _rewrite_aln(src_text: str, offset: float) -> tuple:
    """Apply offset to the AlphaOffset header and every data row's TILT
    column (last of 10 whitespace-separated fields -- see parsers.py's
    parse_aln_file, which parses purely by field count/position, not fixed
    width, so reformatting the row is safe). Returns (new_text, n_rows)."""
    out_lines = []
    n_rows = 0
    for line in src_text.splitlines(keepends=True):
        stripped = line.strip()

        m = _ALPHA_RE.match(stripped)
        if m:
            new_alpha = float(m.group(2)) + offset
            out_lines.append(f'# AlphaOffset = {new_alpha:8.2f}\n')
            continue

        if stripped and not stripped.startswith('#'):
            parts = stripped.split()
            if len(parts) == 10:
                try:
                    tilt = float(parts[-1]) + offset
                    parts[-1] = f'{tilt:.2f}'
                    out_lines.append('  ' + '  '.join(parts) + '\n')
                    n_rows += 1
                    continue
                except ValueError:
                    pass  # not actually a data row -- fall through unchanged

        out_lines.append(line)

    return ''.join(out_lines), n_rows


def _rewrite_imod_tlt(src_text: str, offset: float) -> tuple:
    out_lines = []
    n_rows = 0
    for line in src_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            try:
                tilt = float(stripped) + offset
                out_lines.append(f'{tilt:9.3f}\n')
                n_rows += 1
                continue
            except ValueError:
                pass
        out_lines.append(line)
    return ''.join(out_lines), n_rows


def _apply_offset(aln_files: list, offset: float, dry_run: bool):
    prefix = '[DRY RUN] ' if dry_run else ''

    already_backed_up = [f for f in aln_files
                         if f.with_suffix(f.suffix + '.bak').exists()]
    if already_backed_up:
        print(f'Note: {len(already_backed_up)} file(s) already have a .bak backup.')
        print('      Offset applied to ORIGINAL .bak angles — no compounding.')
        print()

    n_with_imod = sum(1 for f in aln_files if _imod_tlt_path(f).exists())

    if not dry_run:
        print('This will:')
        print(f'  1. Back up each ts-xxx.aln (and ts-xxx_Imod/ts-xxx_st.tlt if present) to .bak')
        print(f'     (skipped where .bak already exists)')
        print(f'  2. Apply {offset:+.4f}° to: AlphaOffset header, TILT column, and the')
        print(f'     matching IMOD _st.tlt file ({n_with_imod}/{len(aln_files)} TS have one)')
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
        new_text, n_rows = _rewrite_aln(src.read_text(), offset)
        print(f'{prefix}Offset : {aln_path.name}  (AlphaOffset header + TILT column, '
              f'{n_rows} tilts, {offset:+.4f}°)')
        if not dry_run:
            aln_path.write_text(new_text)

        imod_path = _imod_tlt_path(aln_path)
        if not imod_path.exists():
            continue
        imod_bak = imod_path.with_suffix(imod_path.suffix + '.bak')
        if not imod_bak.exists():
            print(f'{prefix}Backup : {imod_path.name} -> {imod_bak.name}')
            if not dry_run:
                shutil.copy2(imod_path, imod_bak)
        else:
            print(f'         Skip backup (exists): {imod_bak.name}')
        imod_src = imod_bak if imod_bak.exists() else imod_path
        new_imod_text, n_imod_rows = _rewrite_imod_tlt(imod_src.read_text(), offset)
        print(f'{prefix}Offset : {imod_path.name}  ({n_imod_rows} tilts, {offset:+.4f}°)')
        if not dry_run:
            imod_path.write_text(new_imod_text)

    print()
    if dry_run:
        print('Dry run complete — no files modified.')
    else:
        print('Done. Originals backed up as .bak files.')


def _restore_backup(aln_files: list, dry_run: bool):
    prefix = '[DRY RUN] ' if dry_run else ''

    pairs = []  # (target, backup) for both .aln and its IMOD .tlt companion
    for f in aln_files:
        bak = f.with_suffix(f.suffix + '.bak')
        if bak.exists():
            pairs.append((f, bak))
        imod_path = _imod_tlt_path(f)
        imod_bak = imod_path.with_suffix(imod_path.suffix + '.bak')
        if imod_bak.exists():
            pairs.append((imod_path, imod_bak))

    if not pairs:
        print('No .bak backup files found — nothing to restore.')
        sys.exit(0)

    if not dry_run:
        print('This will:')
        print(f'  Overwrite {len(pairs)} file(s) with their .bak originals')
        print(f'  The .bak files will be removed after a successful restore')
        print(f'  Directory: {aln_files[0].parent}')
        print()
        if not _confirm('Proceed?'):
            print('Aborted.')
            sys.exit(0)
        print()

    for target, bak in pairs:
        print(f'{prefix}Restore: {bak.name} -> {target.name}')
        if not dry_run:
            shutil.copy2(bak, target)
            bak.unlink()

    print()
    if dry_run:
        print(f'Dry run complete — {len(pairs)} file(s) would be restored.')
    else:
        print(f'Done. {len(pairs)} file(s) restored; .bak files removed.')


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
