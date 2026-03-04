"""
enrich — populate project.json with reference data.

This command is the canonical way to register reference data that downstream
commands rely on for auto-fill.  It is also an escape hatch for datasets
processed outside the standard pipeline.

The normal pipeline populates these sections automatically:
  validate-mdoc        → mdoc_data      (--mdoc-data)
  run-aretomo3 --cmd 0 → input_stacks   (--mrc-data, --tlt-data)
  analyse (first run)  → lamella_assignments (--lamellae)

Use enrich when:
  - data was processed externally and the sections are missing, OR
  - you want to force-overwrite an existing section (--force).

Sections
--------
  mdoc_data            per-frame metadata (angpix, dose, stage position, …)
                       Required by analyse for stage position plots and enrichment.

  input_stacks.stacks  paths and dimensions of ts-*.mrc stacks
                       Required by run-aretomo3 --cmd 2 to locate MRC files.

  input_stacks.tlt_dir directory containing ts-xxx_TLT.txt files
                       Required by analyse for dose, z_value, stage positions.

  lamella_assignments  ts-name → lamella cluster mapping
                       Locks clustering so repeated analyse runs are consistent.

Typical usage
-------------
  # Register everything for a manually processed dataset
  aretomo3-preprocess enrich \\
      --mdoc-data frames/ \\
      --mrc-data  run001/ \\
      --tlt-data  run001/ \\
      --lamellae  run001_analysis/lamella_positions.csv

  # Re-register mdoc data after re-running validate-mdoc
  aretomo3-preprocess enrich --mdoc-data frames/ --force
"""

import csv as _csv_module
import sys
import datetime
from pathlib import Path
import argparse

try:
    import mdocfile as _mdocfile
    _HAS_MDOCFILE = True
except ImportError:
    _HAS_MDOCFILE = False

from aretomo3_preprocess.shared.project_json import (
    load as _load_project, update_section,
)
from aretomo3_preprocess.shared.project_state import register_input_stacks


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_mdoc_data(frames_dir: Path, force: bool):
    """Parse all *.mdoc files and write mdoc_data to project.json."""
    if not _HAS_MDOCFILE:
        print('  ERROR: mdocfile not installed — cannot parse mdoc files')
        print('         Install with: pip install mdocfile')
        return

    existing = _load_project().get('mdoc_data', {}).get('per_ts')
    if existing and not force:
        print(f'  mdoc_data already registered ({len(existing)} TS).')
        print(f'  Use --force to overwrite.')
        return

    from aretomo3_preprocess.shared.parsers import parse_mdoc_file

    mdoc_files = sorted(frames_dir.glob('*.mdoc'))
    if not mdoc_files:
        print(f'  ERROR: no .mdoc files found in {frames_dir}')
        return

    prior = _load_project().get('mdoc_data', {}).get('per_ts', {})
    new_entries = {}
    n_ok = n_fail = 0
    for path in mdoc_files:
        try:
            mdoc_data, angpix = parse_mdoc_file(path)
        except Exception as exc:
            print(f'    FAIL  {path.name}: {exc}')
            n_fail += 1
            continue
        if mdoc_data:
            new_entries[path.stem] = {
                'angpix':  angpix,
                'frames':  {str(k): v for k, v in mdoc_data.items()},
            }
            n_ok += 1
        else:
            n_fail += 1

    if not new_entries:
        print(f'  ERROR: no mdoc data extracted from {frames_dir}')
        return

    merged = {**prior, **new_entries}
    update_section('mdoc_data', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_ts':      len(merged),
        'per_ts':    merged,
    })
    print(f'  mdoc_data: {n_ok} TS parsed'
          + (f' ({len(merged)} total in project.json)' if prior else ''))
    if n_fail:
        print(f'  {n_fail} files could not be parsed')


def _enrich_mrc_data(mrc_dir: Path, in_skips: list, force: bool):
    """Scan mrc_dir for ts-*.mrc stacks and register them in project.json."""
    existing = _load_project().get('input_stacks', {}).get('stacks')
    if existing and not force:
        print(f'  input_stacks already registered ({len(existing)} stacks).')
        print(f'  Use --force to overwrite.')
        return
    register_input_stacks(mrc_dir, in_skips=in_skips)


def _enrich_tlt_data(tlt_dir: Path, force: bool):
    """Register tlt_dir (directory with _TLT.txt files) in project.json."""
    existing = _load_project().get('input_stacks', {}).get('tlt_dir')
    if existing and not force:
        print(f'  tlt_dir already registered: {existing}')
        print(f'  Use --force to overwrite.')
        return

    tlt_files = list(tlt_dir.glob('*_TLT.txt'))
    if not tlt_files:
        print(f'  ERROR: no _TLT.txt files found in {tlt_dir}')
        return

    # Merge into existing input_stacks section (preserve stacks, cmd0_outdir, etc.)
    proj             = _load_project()
    section          = dict(proj.get('input_stacks', {}))
    section['tlt_dir']   = str(tlt_dir.resolve())
    section['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    update_section('input_stacks', section)
    print(f'  tlt_dir: {tlt_dir.resolve()}  ({len(tlt_files)} _TLT.txt files)')


def _enrich_lamellae(csv_path: Path, force: bool):
    """Load lamella_positions.csv and write lamella_assignments to project.json."""
    existing = _load_project().get('lamella_assignments', {}).get('positions')
    if existing and not force:
        print(f'  lamella_assignments already registered ({len(existing)} TS).')
        print(f'  Use --force to overwrite.')
        return

    positions = {}
    with open(csv_path, newline='') as fh:
        for row in _csv_module.DictReader(fh):
            ts_name = row.get('ts_name', '').strip()
            lamella = row.get('lamella', '').strip()
            if ts_name and lamella:
                try:
                    positions[ts_name] = int(lamella)
                except ValueError:
                    pass

    if not positions:
        print(f'  ERROR: no lamella assignments found in {csv_path}')
        print(f'         Expected columns: ts_name, lamella')
        return

    update_section('lamella_assignments', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_ts':      len(positions),
        'positions': positions,
    })
    print(f'  lamella_assignments: {len(positions)} TS registered from {csv_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'enrich',
        help='Populate project.json with reference data (mdoc, MRC, TLT, lamellae)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--mdoc-data', default=None, metavar='DIR',
                   help='Directory containing ts-*.mdoc (or Position_*.mdoc) files.  '
                        'Populates mdoc_data in project.json '
                        '(angpix, dose, stage position per frame).  '
                        'Normally populated automatically by validate-mdoc.')
    p.add_argument('--mrc-data', default=None, metavar='DIR',
                   help='Directory containing ts-*.mrc stacks.  '
                        'Populates input_stacks.stacks in project.json '
                        '(path, nx, ny, nz, angpix).  '
                        'Normally populated automatically by run-aretomo3 --cmd 0.')
    p.add_argument('--tlt-data', default=None, metavar='DIR',
                   help='Directory containing ts-xxx_TLT.txt files (the cmd=0 '
                        'output directory).  Populates input_stacks.tlt_dir in '
                        'project.json.  '
                        'Normally populated automatically by run-aretomo3 --cmd 0.')
    p.add_argument('--lamellae', default=None, metavar='CSV',
                   help='lamella_positions.csv from a previous analyse run.  '
                        'Populates lamella_assignments in project.json '
                        '(ts-name → lamella cluster).  '
                        'Normally populated automatically by the first analyse run.')
    p.add_argument('--in-skips', nargs='*', metavar='PATTERN',
                   default=['_CTF', '_Vol', '_EVN', '_ODD'],
                   help='Stem substrings to exclude when scanning --mrc-data '
                        '(default: _CTF _Vol _EVN _ODD).')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing data in project.json.  '
                        'Without --force, enrich skips sections that are already '
                        'populated and prints a message.')
    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    did_anything = False

    if args.mdoc_data is not None:
        frames_dir = Path(args.mdoc_data)
        if not frames_dir.is_dir():
            print(f'ERROR: --mdoc-data {frames_dir} is not a directory')
            sys.exit(1)
        print(f'Populating mdoc_data from {frames_dir}/')
        _enrich_mdoc_data(frames_dir, args.force)
        did_anything = True

    if args.mrc_data is not None:
        mrc_dir = Path(args.mrc_data)
        if not mrc_dir.is_dir():
            print(f'ERROR: --mrc-data {mrc_dir} is not a directory')
            sys.exit(1)
        print(f'Registering MRC stacks from {mrc_dir}/')
        _enrich_mrc_data(mrc_dir, in_skips=args.in_skips, force=args.force)
        did_anything = True

    if args.tlt_data is not None:
        tlt_dir = Path(args.tlt_data)
        if not tlt_dir.is_dir():
            print(f'ERROR: --tlt-data {tlt_dir} is not a directory')
            sys.exit(1)
        print(f'Registering TLT dir from {tlt_dir}/')
        _enrich_tlt_data(tlt_dir, args.force)
        did_anything = True

    if args.lamellae is not None:
        csv_path = Path(args.lamellae)
        if not csv_path.exists():
            print(f'ERROR: --lamellae {csv_path} not found')
            sys.exit(1)
        print(f'Loading lamella assignments from {csv_path}')
        _enrich_lamellae(csv_path, args.force)
        did_anything = True

    if not did_anything:
        print('ERROR: at least one of --mdoc-data, --mrc-data, --tlt-data, '
              '--lamellae must be given.')
        sys.exit(1)
