"""
project_state.py — high-level accessors for aretomo3_project.json

Answers "what do we know about this project?" from data already stored
by earlier pipeline stages.  Used by commands as fallback defaults so
users can omit redundant path arguments on re-runs.

All functions load the project file from the current working directory
and return None (not an exception) when the information is absent.
"""

from __future__ import annotations

import csv as _csv_module
import datetime
from pathlib import Path
from typing import Optional, Set

from aretomo3_preprocess.shared.project_json import load as _load, update_section


# ─────────────────────────────────────────────────────────────────────────────
# Read-only accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_frames_dir() -> Optional[Path]:
    """Return the frames directory recorded by rename-ts, or None."""
    data = _load()
    value = data.get('rename_ts', {}).get('input')
    return Path(value) if value else None


def resolve_original_mdoc_path(rename_lookup: dict, ts_name: str) -> Optional[Path]:
    """
    Look up ts_name's original (pre-rename) mdoc Path in an already-loaded
    rename_ts.lookup dict ({'ts-XXXX.mdoc': original_absolute_path}), or
    None if not found. Pure lookup, no project.json I/O -- for callers
    that already have `project`/`rename_ts.lookup` loaded (e.g. in a loop
    over many TS) and want to avoid re-reading it per TS.
    """
    value = rename_lookup.get(f'{ts_name}.mdoc')
    return Path(value) if value else None


def get_ts_to_original_stem() -> dict:
    """
    {renamed ts-XXXX stem -> original (pre-rename) filename stem, e.g.
    'Position_1'}, built from rename_ts.lookup (ts-XXXX.mdoc -> original
    absolute path) in project.json. Empty dict if rename-ts hasn't been
    run yet.

    mdoc_data.per_ts is keyed by the ORIGINAL stem, not the renamed
    ts-XXXX name (mdocs are validated before rename-ts in the documented
    pipeline order, so that's the natural key; .aln/volumes are named
    ts-XXXX). Use this to translate a ts-XXXX name into the mdoc_data key
    to read -- or, when saving, to normalize a ts-XXXX.mdoc symlink's own
    filename stem back to the same original key its source Position_N.mdoc
    would use, so re-validating post-rename updates one entry instead of
    creating a second one under the ts-XXXX name.
    """
    lookup = _load().get('rename_ts', {}).get('lookup', {})
    return {
        Path(ts_mdoc).stem: Path(orig_path).stem
        for ts_mdoc, orig_path in lookup.items()
    }


def get_angpix() -> Optional[float]:
    """
    Return the most common pixel size from mdoc_data.per_ts,
    or from analyse.global_suggested.angpix as fallback.
    Returns None if not available.
    """
    data = _load()
    per_ts = data.get('mdoc_data', {}).get('per_ts', {})
    if per_ts:
        vals = [ts.get('angpix') for ts in per_ts.values()
                if ts.get('angpix') is not None]
        if vals:
            # mode: most common value
            from collections import Counter
            return Counter(vals).most_common(1)[0][0]
    # fallback: analyse.global_suggested.angpix
    return data.get('analyse', {}).get('global_suggested', {}).get('angpix')


def get_calibrated_apix() -> Optional[float]:
    """
    Return the confirmed/calibrated pixel size, if one has been recorded --
    either up front via `validate-mdoc --calibrated-apix`, or lazily by any
    command's real (non-dry-run) preflight after a --apix/--angpix value
    has passed validation (matched cleanly, or was --force'd through).

    This is deliberately separate from the raw mdoc PixelSpacing (which is
    uncalibrated and routinely *expected* to disagree with the real value):
    once a calibrated value exists, it's the reference every later
    preflight check should compare against instead of the raw mdoc, so a
    deliberately-confirmed override doesn't require --force again on every
    subsequent run. Returns None if nothing has been recorded yet.
    """
    return _load().get('calibrated_apix', {}).get('value')


def record_calibrated_apix(value: float, source: str) -> None:
    """Record the confirmed/calibrated pixel size, so later preflight
    checks (any command) compare against it instead of the raw,
    uncalibrated mdoc PixelSpacing. See get_calibrated_apix()."""
    update_section('calibrated_apix', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'value':     value,
        'source':    source,
    })


def resolve_reference_apix() -> tuple:
    """
    Best available "ground truth" pixel size for a --apix/--angpix
    mismatch check, for callers with no raw .mdoc file directly in scope
    (--cmd 1/2, run-aretomo3-per-ts, analyse). Priority:
      1. calibrated_apix   -- see get_calibrated_apix()
      2. mdoc_data.per_ts  -- project.json, from validate-mdoc (uncalibrated)
    Returns (value, label); value is None if neither is available.
    """
    calibrated = get_calibrated_apix()
    if calibrated is not None:
        return calibrated, 'project.json (calibrated_apix, previously confirmed)'
    ps = get_angpix()
    if ps is not None:
        return ps, 'project.json (mdoc_data, from validate-mdoc)'
    return None, None


def get_handedness() -> Optional[dict]:
    """
    Return the recorded physical (volume) handedness determination, if
    `pytom-ribo-auto --check-handedness` has confirmed one for this
    project -- {'mirror': bool, 'particle', 'per_ts', 'timestamp',
    'source'} -- or None if it hasn't been run yet.

    This is the *physical/volume* handedness (whether the reconstructed
    tomogram is a mirror-image of reality, corrected via --mirror on the
    picking template) -- unrelated to RELION's separate "defocus
    handedness" concept (whether defocus increases/decreases with Z).
    """
    return _load().get('handedness') or None


def record_handedness(mirror: bool, particle: str, per_ts: dict, source: str) -> None:
    """Record a --check-handedness determination so later pytom-ribo-auto
    runs (or a human) don't need to re-derive it from scratch. See
    get_handedness()."""
    update_section('handedness', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'mirror':    mirror,
        'particle':  particle,
        'per_ts':    per_ts,
        'source':    source,
    })


def get_voltage() -> Optional[float]:
    """
    Return the most common accelerating voltage (kV) from mdoc_data.per_ts,
    or None if not available (mdoc_data.per_ts not yet populated -- e.g.
    before enrich/validate-mdoc has run -- or acquisition.voltage missing
    from an older mdoc parse that predates that field).
    """
    per_ts = _load().get('mdoc_data', {}).get('per_ts', {})
    vals = [ts.get('acquisition', {}).get('voltage') for ts in per_ts.values()
            if ts.get('acquisition', {}).get('voltage') is not None]
    if not vals:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def get_run_params() -> Optional[dict]:
    """
    Return the microscope params ({'kv', 'cs', 'amp_contrast', 'apix',
    'cmd', 'timestamp'}) recorded by the most recent real (non-dry-run)
    run-aretomo3 invocation for this project, or None if run-aretomo3
    hasn't recorded any yet.

    This is the self-consistency baseline for --cs/--amp-contrast, which
    (unlike --apix/--kv) have no source in the mdoc to check against --
    SerialEM doesn't log spherical aberration or amplitude contrast, so
    the only "ground truth" available is whatever value was actually used
    on a prior stage of the same pipeline.
    """
    return _load().get('run_aretomo3_params') or None


def record_run_params(kv: float, cs: float, amp_contrast: float,
                      apix: float, cmd: int) -> None:
    """Record the microscope params used by a real run-aretomo3 invocation,
    so later stages (cmd=1/2) can be checked for self-consistency against
    them via get_run_params()."""
    update_section('run_aretomo3_params', {
        'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
        'kv':           kv,
        'cs':           cs,
        'amp_contrast': amp_contrast,
        'apix':         apix,
        'cmd':          cmd,
    })


def get_latest_analysis_dir() -> Optional[Path]:
    """Return the output directory from the last analyse run, or None."""
    data = _load()
    value = data.get('analyse', {}).get('output_dir')
    return Path(value) if value else None


def get_cmd0_outdir() -> Optional[Path]:
    """Return the cmd=0 output directory recorded in input_stacks, or None."""
    data = _load()
    value = data.get('input_stacks', {}).get('cmd0_outdir')
    return Path(value) if value else None


def get_tlt_dir() -> Optional[Path]:
    """
    Return the directory containing _TLT.txt files, or None.

    Reads input_stacks.tlt_dir (saved by cmd=0 run-aretomo3), with fallback
    to input_stacks.cmd0_outdir (same location, older project files).
    """
    data = _load()
    stored = data.get('input_stacks', {})
    value = stored.get('tlt_dir') or stored.get('cmd0_outdir')
    return Path(value) if value else None


def get_input_stacks() -> Optional[dict]:
    """
    Return the stacks dict from input_stacks, or None.
    Maps ts_name -> {path, nx, ny, nz, angpix}.
    """
    data = _load()
    return data.get('input_stacks', {}).get('stacks') or None


def get_gain_check_dir() -> Optional[Path]:
    """Return the gain-check output directory from project.json, or None."""
    data = _load()
    value = data.get('gain_check', {}).get('output_dir')
    return Path(value) if value else None


def get_defocus_data() -> Optional[dict]:
    """Return the defocus_data.per_ts dict {ts_name: ref_defocus_um}, or None."""
    data = _load()
    return data.get('defocus_data', {}).get('per_ts') or None


def resolve_selected_ts(csv_path: Optional[str] = None) -> Optional[Set[str]]:
    """
    Load the TS selection set from a ts_selection.csv file.

    Returns the set of selected TS names (selected==1), or None if
    csv_path is None or the file cannot be read.
    """
    if csv_path is None:
        return None
    p = Path(csv_path)
    if not p.exists():
        print(f'WARNING: --select-ts {p} not found; processing all TS')
        return None
    selected = set()
    with open(p) as fh:
        reader = _csv_module.DictReader(fh)
        for row in reader:
            if row.get('selected', '').strip() == '1':
                selected.add(row['ts_name'])
    n = len(selected)
    print(f'TS selection: {n} selected from {p}')
    return selected if selected else None


# ─────────────────────────────────────────────────────────────────────────────
# Stack registration (replaces _register_cmd0_stacks / _save_stacks_to_project)
# ─────────────────────────────────────────────────────────────────────────────

def register_input_stacks(out_dir: Path, in_skips: list = None,
                          tlt_dir: Path = None):
    """
    Scan out_dir for ts-*.mrc files and register them in project.json
    under 'input_stacks'.

    Replaces _register_cmd0_stacks (run_aretomo3.py) and
    _save_stacks_to_project (run_aretomo3_per_ts.py).

    Parameters
    ----------
    out_dir   : Path  Directory to scan for ts-*.mrc stacks
    in_skips  : list  Stem substrings to exclude (e.g. ['_Vol', '_CTF', '_EVN', '_ODD'])
    tlt_dir   : Path  Directory containing _TLT.txt files (cmd=0 output dir);
                      saved so that analyse can find them automatically.
    """
    try:
        import mrcfile
    except ImportError:
        mrcfile = None

    skips = [s for s in (in_skips or []) if s]
    all_mrc = sorted(out_dir.glob('ts-*.mrc'))
    stack_files = [f for f in all_mrc
                   if not any(s in f.stem for s in skips)]
    if not stack_files:
        return

    stacks = {}
    for f in stack_files:
        info = {'path': str(f.resolve())}
        if mrcfile is not None:
            try:
                with mrcfile.mmap(f, mode='r', permissive=True) as m:
                    info.update({
                        'nx':     int(m.header.nx),
                        'ny':     int(m.header.ny),
                        'nz':     int(m.header.nz),
                        'angpix': round(float(m.voxel_size.x), 4),
                    })
            except Exception:
                pass
        stacks[f.stem] = info

    values = {
        'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
        'cmd0_outdir': str(out_dir.resolve()),
        'n_stacks':    len(stacks),
        'stacks':      stacks,
    }
    if tlt_dir is not None:
        values['tlt_dir'] = str(tlt_dir.resolve())
    else:
        # Preserve existing tlt_dir so it is not lost when stacks are re-registered
        existing_tlt = _load().get('input_stacks', {}).get('tlt_dir')
        if existing_tlt:
            values['tlt_dir'] = existing_tlt

    update_section(section='input_stacks', values=values)
    print(f'Registered {len(stacks)} input stacks in project.json  [input_stacks]')
    if tlt_dir is not None:
        print(f'Registered TLT directory     in project.json  [input_stacks.tlt_dir]')


# ─────────────────────────────────────────────────────────────────────────────
# Stack loader (replaces _load_stacks_from_project in run_aretomo3_per_ts.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_input_stacks() -> tuple:
    """
    Load the input_stacks section from project.json in the current directory.

    Returns (mrc_files, source_info) where:
      mrc_files   — list of Path objects (only paths that exist on disk)
      source_info — dict with 'cmd0_outdir', 'timestamp', 'n_registered', 'n_found'
    Returns (None, None) if the section is absent.
    """
    data = _load()
    stored = data.get('input_stacks', {})
    if not stored or not stored.get('stacks'):
        return None, None

    mrc_files = []
    for ts_name in sorted(stored['stacks']):
        info = stored['stacks'][ts_name]
        p = Path(info['path'])
        if p.exists():
            mrc_files.append(p)

    source_info = {
        'cmd0_outdir':  stored.get('cmd0_outdir', '?'),
        'timestamp':    stored.get('timestamp', '?'),
        'n_registered': stored.get('n_stacks', len(stored['stacks'])),
        'n_found':      len(mrc_files),
        'stacks':       stored['stacks'],
    }
    return mrc_files, source_info
