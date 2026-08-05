"""
enrich — populate project.json with reference data.

This command is the canonical way to register reference data that downstream
commands rely on for auto-fill.  It is also an escape hatch for datasets
processed outside the standard pipeline.

The normal pipeline populates these sections automatically:
  validate-mdoc        → mdoc_data             (--mdoc-data)
  run-aretomo3 --cmd 0 → input_stacks          (--mrc-data, --tlt-data)
  run-aretomo3 --cmd 0 → frame_lookup          (--frame-lookup)
  analyse (first run)  → lamella_assignments   (--lamellae)

Note: there is no defocus_data section / --defocus-data option. Per-TS
reference defocus for imod-mtffilter is read fresh from its own --input
directory's ts-*_CTF.txt every run, not cached here -- see
imod_mtffilter.py's docstring and shared/parsers.py:compute_reference_defocus.
An earlier project.json-cached defocus_data section was removed for
exactly the staleness reason that pattern avoids: defocus estimates
change whenever AreTomo3/CTFFIND is re-run, and a cache has no way to know
it's gone stale.

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

  frame_lookup         per-TS SEC <-> acq_order/z_value bridge + dark-SEC
                       flags, from ts-*.aln + ts-*_TLT.txt. The canonical
                       cross-referencing table -- see CLAUDE.md's
                       "frame_lookup" section and shared/project_state.py's
                       resolve_frame()/get_frame_lookup().

  lamella_assignments  ts-name → lamella cluster mapping
                       Locks clustering so repeated analyse runs are consistent.

Typical usage
-------------
  # Register everything for a manually processed dataset
  aretomo3-preprocess enrich \\
      --mdoc-data   frames/ \\
      --mrc-data    run001/ \\
      --tlt-data    run001/ \\
      --frame-lookup run001/ \\
      --lamellae    run001_analysis/lamella_positions.csv

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
from aretomo3_preprocess.shared.project_state import (
    register_input_stacks, record_tool_path, register_mdoc_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_mdoc_data(frames_dir: Path, force: bool):
    """Parse mdoc files and write mdoc_data to project.json.

    Prefers ts-*.mdoc (the rename-ts symlinks -- exactly the curated set
    that's actually used downstream) over a raw *.mdoc glob when both
    exist in frames_dir. rename-ts creates its symlinks alongside the
    original Position_*.mdoc files it points at, so a bare *.mdoc glob
    there double-counts every renamed TS once under its original stem and
    once under its ts-XXX symlink name (e.g. 172 real TS -> 344 mdoc
    files). Each ts-XXX.mdoc entry is still keyed by its RESOLVED
    (original) stem, not 'ts-XXX' -- matching the existing convention
    that mdoc_data.per_ts is keyed by original stem (see
    get_ts_to_original_stem()'s docstring), so this is a pure dedup, not
    a change to how downstream code looks entries up. Falls back to a raw
    *.mdoc glob when no ts-*.mdoc symlinks exist yet (e.g. this is being
    run before rename-ts in the pipeline).
    """
    if not _HAS_MDOCFILE:
        print('  ERROR: mdocfile not installed — cannot parse mdoc files')
        print('         Install with: pip install mdocfile')
        return False

    existing = _load_project().get('mdoc_data', {}).get('per_ts')
    if existing and not force:
        print(f'  mdoc_data already registered ({len(existing)} TS).')
        print(f'  Use --force to overwrite.')
        return True

    ts_mdocs = sorted(frames_dir.glob('ts-*.mdoc'))
    if ts_mdocs:
        mdoc_files = ts_mdocs
        print(f'  Found {len(ts_mdocs)} ts-*.mdoc symlinks — reading only the '
              f'renamed/curated set (not every raw *.mdoc in this directory).')
    else:
        mdoc_files = sorted(frames_dir.glob('*.mdoc'))
    if not mdoc_files:
        print(f'  ERROR: no .mdoc files found in {frames_dir}')
        return False

    result = register_mdoc_data(mdoc_files)
    if result['n_ok'] == 0:
        print(f'  ERROR: no mdoc data extracted from {frames_dir}')
        return False
    print(f'  mdoc_data: {result["n_ok"]} TS parsed'
          + (f' ({result["merged_count"]} total in project.json)'
             if result['merged_count'] > result['n_ok'] else ''))
    if result['n_fail']:
        print(f'  {result["n_fail"]} files could not be parsed')
    return True


def _enrich_mrc_data(mrc_dir: Path, in_skips: list, force: bool):
    """Scan mrc_dir for ts-*.mrc stacks and register them in project.json."""
    existing = _load_project().get('input_stacks', {}).get('stacks')
    if existing and not force:
        print(f'  input_stacks already registered ({len(existing)} stacks).')
        print(f'  Use --force to overwrite.')
        return True
    found = register_input_stacks(mrc_dir, in_skips=in_skips)
    if not found:
        print(f'  ERROR: no ts-*.mrc stacks found in {mrc_dir} (after --in-skips filtering)')
    return found


def _enrich_tlt_data(tlt_dir: Path, force: bool):
    """Register tlt_dir (directory with _TLT.txt files) in project.json."""
    existing = _load_project().get('input_stacks', {}).get('tlt_dir')
    if existing and not force:
        print(f'  tlt_dir already registered: {existing}')
        print(f'  Use --force to overwrite.')
        return True

    tlt_files = list(tlt_dir.glob('*_TLT.txt'))
    if not tlt_files:
        print(f'  ERROR: no _TLT.txt files found in {tlt_dir}')
        return False

    # Merge into existing input_stacks section (preserve stacks, cmd0_outdir, etc.)
    proj             = _load_project()
    section          = dict(proj.get('input_stacks', {}))
    section['tlt_dir']   = str(tlt_dir.resolve())
    section['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    update_section('input_stacks', section)
    print(f'  tlt_dir: {tlt_dir.resolve()}  ({len(tlt_files)} _TLT.txt files)')
    return True


def _enrich_frame_lookup(aln_dir: Path, force: bool):
    """Register frame_lookup from aln_dir (see project_state.register_frame_lookup).
    Thin wrapper adding enrich's usual already-populated/--force check --
    register_frame_lookup itself always merges (safe to call repeatedly
    from run-aretomo3's own auto-fill as TS complete incrementally), so
    this is enrich's manual-invocation guard, not a change to that
    function's own merge behaviour."""
    existing = _load_project().get('frame_lookup', {}).get('per_ts')
    if existing and not force:
        print(f'  frame_lookup already registered ({len(existing)} TS).')
        print(f'  Use --force to overwrite/refresh.')
        return True

    from aretomo3_preprocess.shared.project_state import register_frame_lookup
    found = register_frame_lookup(aln_dir)
    if not found:
        print(f'  ERROR: no ts-*.aln + matching ts-*_TLT.txt pairs found in {aln_dir}')
    return found


def _enrich_lamellae(csv_path: Path, force: bool):
    """Load lamella_positions.csv and write lamella_assignments to project.json."""
    existing = _load_project().get('lamella_assignments', {}).get('positions')
    if existing and not force:
        print(f'  lamella_assignments already registered ({len(existing)} TS).')
        print(f'  Use --force to overwrite.')
        return True

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
        return False

    update_section('lamella_assignments', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_ts':      len(positions),
        'positions': positions,
    })
    print(f'  lamella_assignments: {len(positions)} TS registered from {csv_path.name}')
    return True


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
    p.add_argument('--frame-lookup', default=None, metavar='DIR',
                   help='Directory containing ts-*.aln + ts-*_TLT.txt files '
                        '(e.g. a run-aretomo3 --cmd 0 output dir).  '
                        'Populates frame_lookup in project.json (per-TS SEC '
                        '<-> acq_order/z_value bridge + dark-SEC flags, see '
                        'CLAUDE.md).  '
                        'Normally populated automatically by run-aretomo3 --cmd 0.')
    p.add_argument('--in-skips', nargs='*', metavar='PATTERN',
                   default=['_CTF', '_Vol', '_EVN', '_ODD'],
                   help='Stem substrings to exclude when scanning --mrc-data '
                        '(default: _CTF _Vol _EVN _ODD).')
    tools = p.add_argument_group('external tool paths')
    tools.add_argument('--set-path-aretomo3', default=None, metavar='PATH',
                       help='Remember an AreTomo3 binary/path for this project '
                            '(run-aretomo3/run-aretomo3-per-ts auto-fill --aretomo3 '
                            'from this instead of needing it every invocation).')
    tools.add_argument('--set-path-pytom', default=None, metavar='DIR',
                       help='Remember a pytom-match-pick bin/ directory for this '
                            'project (pytom-match/pytom-ribo-auto auto-fill '
                            '--pytom-dir from this).')
    tools.add_argument('--set-path-imod', default=None, metavar='DIR',
                       help='Remember an IMOD bin/ directory for this project '
                            '(pytom-ribo-auto auto-fills --imod-bin-dir from this).')
    tools.add_argument('--set-path-gapstop', default=None, metavar='DIR',
                       help='Remember a gapstop env/ directory for this project '
                            '(gapstop-match auto-fills --gapstop-dir from this).')
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
    # Distinct from did_anything: did_anything just means "at least one
    # section was requested" (true even if every handler failed to find
    # anything). all_ok tracks whether every REQUESTED handler actually
    # succeeded -- a handler that prints its own "ERROR: nothing found"
    # and returns False used to still count as done_anything=True, so
    # `enrich --mdoc-data empty_dir/` exited 0 despite registering
    # nothing, giving a scripted pipeline chain no signal.
    all_ok = True

    if args.mdoc_data is not None:
        frames_dir = Path(args.mdoc_data)
        if not frames_dir.is_dir():
            print(f'ERROR: --mdoc-data {frames_dir} is not a directory')
            sys.exit(1)
        print(f'Populating mdoc_data from {frames_dir}/')
        all_ok &= bool(_enrich_mdoc_data(frames_dir, args.force))
        did_anything = True

    if args.mrc_data is not None:
        mrc_dir = Path(args.mrc_data)
        if not mrc_dir.is_dir():
            print(f'ERROR: --mrc-data {mrc_dir} is not a directory')
            sys.exit(1)
        print(f'Registering MRC stacks from {mrc_dir}/')
        all_ok &= bool(_enrich_mrc_data(mrc_dir, in_skips=args.in_skips, force=args.force))
        did_anything = True

    if args.tlt_data is not None:
        tlt_dir = Path(args.tlt_data)
        if not tlt_dir.is_dir():
            print(f'ERROR: --tlt-data {tlt_dir} is not a directory')
            sys.exit(1)
        print(f'Registering TLT dir from {tlt_dir}/')
        all_ok &= bool(_enrich_tlt_data(tlt_dir, args.force))
        did_anything = True

    if args.frame_lookup is not None:
        aln_dir = Path(args.frame_lookup)
        if not aln_dir.is_dir():
            print(f'ERROR: --frame-lookup {aln_dir} is not a directory')
            sys.exit(1)
        print(f'Registering frame lookup from {aln_dir}/')
        all_ok &= bool(_enrich_frame_lookup(aln_dir, args.force))
        did_anything = True

    if args.lamellae is not None:
        csv_path = Path(args.lamellae)
        if not csv_path.exists():
            print(f'ERROR: --lamellae {csv_path} not found')
            sys.exit(1)
        print(f'Loading lamella assignments from {csv_path}')
        all_ok &= bool(_enrich_lamellae(csv_path, args.force))
        did_anything = True

    for tool_name, cli_flag in (('aretomo3', args.set_path_aretomo3),
                                ('pytom',    args.set_path_pytom),
                                ('imod',     args.set_path_imod),
                                ('gapstop',  args.set_path_gapstop)):
        if cli_flag is not None:
            record_tool_path(tool_name, cli_flag)
            print(f'Recorded tool path: {tool_name} -> {cli_flag}')
            did_anything = True

    if not did_anything:
        print('ERROR: at least one of --mdoc-data, --mrc-data, --tlt-data, '
              '--frame-lookup, --lamellae, --set-path-* must be given.')
        sys.exit(1)

    if not all_ok:
        print('\nERROR: one or more requested sections found nothing to register '
              '(see ERROR lines above).')
        sys.exit(1)
