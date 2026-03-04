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


def resolve_selected_ts(csv_path: Optional[str] = None) -> Optional[Set[str]]:
    """
    Load the TS selection set from a ts_selection.csv file.

    Returns the set of selected TS names (selected==1), or None if no
    csv_path is given or the file cannot be read.
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
