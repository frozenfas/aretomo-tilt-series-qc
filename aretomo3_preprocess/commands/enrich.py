"""
enrich — populate project.json with metadata for users who did not follow
the standard pipeline (e.g. imported data from an external AreTomo3 run).

This command is an escape hatch for situations where validate-mdoc and
run-aretomo3 cmd=0 were not used.  It writes the two project.json sections
that downstream commands rely on for auto-filling arguments:

  mdoc_data.per_ts    — per-frame metadata (angpix, dose, stage position, …)
                        Used by analyse (angpix, mdoc enrichment) and
                        as auto-fill source for --apix in other commands.

  input_stacks        — paths and dimensions of ts-*.mrc stacks
                        Used by run-aretomo3-per-ts to find stacks without
                        requiring --mrcdir.

Both sections can be filled independently; omit --frames to skip mdoc_data,
omit --mrcdir to skip input_stacks.

Typical usage
-------------
  # Fill both from a frames directory and run001 output
  aretomo3-preprocess enrich --frames frames/ --mrcdir run001/

  # Fill only mdoc metadata (mrcdir already registered)
  aretomo3-preprocess enrich --frames frames/

  # Fill only input_stacks (mdoc already registered)
  aretomo3-preprocess enrich --mrcdir run001/
"""

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
# mdoc_data population
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_mdoc(frames_dir: Path):
    """Parse all ts-*.mdoc (and *.mdoc) files and write mdoc_data to project.json."""
    if not _HAS_MDOCFILE:
        print('WARNING: mdocfile not installed — cannot parse mdoc files')
        print('         Install with: pip install mdocfile')
        return

    from aretomo3_preprocess.shared.parsers import parse_mdoc_file

    mdoc_files = sorted(frames_dir.glob('*.mdoc'))
    if not mdoc_files:
        print(f'No .mdoc files found in {frames_dir}')
        return

    existing = _load_project().get('mdoc_data', {}).get('per_ts', {})

    new_entries = {}
    n_ok = n_fail = 0
    for path in mdoc_files:
        try:
            mdoc_data, angpix = parse_mdoc_file(path)
        except Exception as exc:
            print(f'  FAIL  {path.name}: {exc}')
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
        print(f'No mdoc data extracted from {frames_dir}')
        return

    merged = {**existing, **new_entries}
    update_section('mdoc_data', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_ts':      len(merged),
        'per_ts':    merged,
    })
    print(f'  mdoc_data: {n_ok} TS parsed'
          + (f' ({len(merged)} total in project.json)' if existing else ''))
    if n_fail:
        print(f'  {n_fail} files could not be parsed')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'enrich',
        help='Populate project.json with mdoc and/or MRC stack metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--frames', '-f', default=None,
                   help='Directory containing ts-*.mdoc (or Position_*.mdoc) '
                        'files.  Populates mdoc_data.per_ts in project.json '
                        '(angpix, dose, stage position per frame).')
    p.add_argument('--mrcdir', '-m', default=None,
                   help='Directory containing ts-*.mrc stacks.  Populates '
                        'input_stacks in project.json (path, nx, ny, nz, angpix).')
    p.add_argument('--in-skips', nargs='*', metavar='PATTERN',
                   default=['_CTF', '_Vol', '_EVN', '_ODD'],
                   help='Stem substrings to exclude from --mrcdir scan '
                        '(default: _CTF _Vol _EVN _ODD).')
    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    if args.frames is None and args.mrcdir is None:
        print('ERROR: at least one of --frames or --mrcdir must be given.')
        sys.exit(1)

    if args.frames is not None:
        frames_dir = Path(args.frames)
        if not frames_dir.is_dir():
            print(f'ERROR: --frames {frames_dir} is not a directory')
            sys.exit(1)
        print(f'Parsing mdoc files from {frames_dir}/')
        _enrich_mdoc(frames_dir)

    if args.mrcdir is not None:
        mrc_dir = Path(args.mrcdir)
        if not mrc_dir.is_dir():
            print(f'ERROR: --mrcdir {mrc_dir} is not a directory')
            sys.exit(1)
        print(f'Registering MRC stacks from {mrc_dir}/')
        register_input_stacks(mrc_dir, in_skips=args.in_skips)
