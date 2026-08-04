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
import re
from pathlib import Path
from typing import Optional, Set

from aretomo3_preprocess.shared.project_json import load as _load, update_section, SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Read-only accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_schema_version() -> int:
    """
    Schema version last stamped into this project.json (see
    project_json.SCHEMA_VERSION) -- i.e. how recent the code was that most
    recently wrote *any* section here. 0 if project.json doesn't exist yet
    or predates this tracking (every write stamps the current version, so a
    project only touched by pre-tracking code has no stamp at all).

    Not a hard compatibility gate -- every accessor in this module already
    reads defensively via .get() and degrades gracefully on missing fields.
    Use this to give a user an honest "re-run X to populate field Y" hint
    instead of guessing from which fields happen to be present.
    """
    return _load().get('project', {}).get('schema_version', 0)


def get_frames_dir() -> Optional[Path]:
    """
    Return the frames directory recorded by rename-ts, or None.

    rename-ts records the input directory per grid
    (rename_ts.grids.<N>.input_dir -- each invocation/grid can use a
    different --input), never a single top-level value. This accessor's
    one caller (run-aretomo3-per-ts's --mdocdir default) only supports a
    single directory for the whole run (it's passed straight through to
    AreTomo3's own -Mdoc flag), so this returns the latest grid's
    input_dir and warns if grids disagree, rather than silently picking
    an arbitrary one. For a genuinely per-TS answer, resolve_frame()'s
    frame_path (via frame_lookup/mdoc_data.frames_dir) is correct even
    across multiple grids -- use that instead when the caller has a
    ts_name in hand.
    """
    data = _load()
    grids = data.get('rename_ts', {}).get('grids', {})
    if not grids:
        return None
    dirs = {g['input_dir'] for g in grids.values() if g.get('input_dir')}
    if not dirs:
        return None
    if len(dirs) > 1:
        print(f'WARNING: get_frames_dir: {len(grids)} grids recorded with '
              f'different input dirs {sorted(dirs)}; using the latest '
              f'grid\'s -- pass --mdocdir explicitly if that\'s wrong')
    latest_grid = max(grids, key=int)
    value = grids[latest_grid].get('input_dir')
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


def resolve_selected_ts(csv_path: Optional[str] = None) -> Optional[Set[str]]:
    """
    Load the TS selection set from a ts-select.csv file.

    Deliberately never auto-loads a previously-recorded selection --
    omitting --select-ts always means "process every TS found", with no
    implicit dependency on an earlier, possibly-forgotten select-ts run
    (e.g. a small test subset from weeks ago silently limiting a full
    production run). Selection only ever applies when --select-ts is
    passed explicitly.

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


# ─────────────────────────────────────────────────────────────────────────────
# Frame lookup: SEC <-> acq_order/z_value bridge (see CLAUDE.md's
# "frame_lookup" section) -- SEC/frame_b/acq_order/z_value are this
# codebase's sanctioned frame cross-referencing keys (never tilt-angle
# proximity, see CLAUDE.md's "Cross-referencing frames"); this table exists
# so that correspondence is derived once, not re-parsed from .aln/_TLT.txt
# independently in every command that needs it.
# ─────────────────────────────────────────────────────────────────────────────

def register_frame_lookup(out_dir: Path) -> None:
    """
    Scan out_dir for ts-*.aln + matching ts-*_TLT.txt files and register
    each TS's SEC <-> z_value/acq_order bridge (+ filename, + which SECs
    are dark) in project.json under 'frame_lookup'.

    The filename (sub_frame_path) and frames_dir are captured HERE, at
    build time, by composing with the already-registered mdoc_data section
    (via get_ts_to_original_stem()) -- not resolved lazily at read time.
    Tried the lazy/composed-at-read-time design first and found it fragile
    in practice: resolve_frame() silently returned sub_frame_path=None
    whenever mdoc_data happened to be registered from a directory other
    than the actual rename-ts symlink dir (e.g. a staging/temp mdoc copy),
    even though frame_lookup's own data was perfectly valid -- a real
    dependency on rename_ts/mdoc_data being correct *every time a frame is
    resolved*, not just once. Baking the filename in at build time makes a
    registered frame_lookup entry self-contained: relion5_convert.py (the
    main filename consumer) needs only this section, not a live mdoc_data
    join on every lookup. If mdoc_data wasn't registered yet when this was
    called, sub_frame_path/frames_dir are None for that run -- re-run (with
    --force via `enrich --frame-lookup`) after registering mdoc_data to
    backfill them.

    frames_dir is mdoc_data.per_ts[stem].frames_dir -- the directory the
    mdoc validated by validate-mdoc actually lived in (recorded there
    because AreTomo3 requires raw movies and their mdoc to be co-located,
    confirmed by rename-ts's own symlinks being created "alongside the
    originals", and now checkable directly via validate-mdoc
    --check-frames). One value per TS, not per frame -- same directory for
    every frame in a TS.

    Each entry also records 'validated': whether _TLT.txt's nominal_tilt +
    that TS's alpha_offset agrees with .aln's TILT column (both frames and
    dark frames) to within 0.05 deg -- see
    parsers.py:check_nominal_tilt_consistency. This is a *validation* of
    the SEC correspondence via tilt-angle agreement, not the correspondence
    mechanism itself (which stays the exact SEC/frame_b keys throughout).

    Safe to call multiple times / on a growing directory (e.g. TS
    processed incrementally) -- merges into any existing per_ts entries
    rather than replacing the whole section.
    """
    from aretomo3_preprocess.shared.parsers import (
        parse_aln_file, parse_tlt_file, check_nominal_tilt_consistency,
    )

    aln_files = sorted(out_dir.glob('ts-*.aln'))
    if not aln_files:
        return

    project = _load()
    existing = project.get('frame_lookup', {}).get('per_ts', {})
    ts_to_stem = get_ts_to_original_stem()
    mdoc_per_ts = project.get('mdoc_data', {}).get('per_ts', {})

    new_entries = {}
    n_ok = n_fail = n_no_filename = 0
    for aln_path in aln_files:
        ts_name = aln_path.stem
        tlt_path = out_dir / f'{ts_name}_TLT.txt'
        if not tlt_path.exists():
            print(f'    SKIP  {ts_name}: no matching {tlt_path.name}')
            n_fail += 1
            continue
        try:
            aln_data = parse_aln_file(aln_path)
            tlt_data = parse_tlt_file(tlt_path)
            if not tlt_data:
                raise ValueError('no rows in _TLT.txt')

            mdoc_entry_for_ts = mdoc_per_ts.get(ts_to_stem.get(ts_name, ''), {})
            mdoc_frames = mdoc_entry_for_ts.get('frames', {})
            frames_dir = mdoc_entry_for_ts.get('frames_dir')
            frames = {}
            for sec, t in tlt_data.items():
                mdoc_entry = mdoc_frames.get(str(t['z_value']))
                frames[str(sec)] = {
                    'z_value':        t['z_value'],
                    'sub_frame_path': mdoc_entry.get('sub_frame_path') if mdoc_entry else None,
                }
            if not any(f['sub_frame_path'] for f in frames.values()):
                n_no_filename += 1

            dark_secs = sorted(df['frame_b'] for df in aln_data['dark_frames'])
            alpha_offset = aln_data.get('alpha_offset') or 0.0

            pairs = [(f['sec'], f['tilt']) for f in aln_data['frames']]
            pairs += [(df['frame_b'], df['tilt']) for df in aln_data['dark_frames']]
            bad = check_nominal_tilt_consistency(pairs, tlt_data, alpha_offset)

            new_entries[ts_name] = {
                'n_total':    aln_data['total_frames'],
                'frames':     frames,
                'dark_secs':  dark_secs,
                'validated':  not bad,
                'frames_dir': frames_dir,
            }
            n_ok += 1
        except Exception as exc:
            print(f'    FAIL  {ts_name}: {exc}')
            n_fail += 1

    if not new_entries:
        return

    merged = {**existing, **new_entries}
    update_section('frame_lookup', {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'per_ts':    merged,
    })
    print(f'Registered frame lookup for {n_ok} TS in project.json  [frame_lookup]'
          + (f'  ({n_fail} skipped)' if n_fail else ''))
    if n_no_filename:
        print(f'  {n_no_filename} TS have no filename data (mdoc_data/rename_ts not yet '
              f'registered) -- re-run enrich --frame-lookup --force after registering them.')


def get_frame_lookup(ts_name: str) -> Optional[dict]:
    """Raw {'n_total', 'frames', 'dark_secs', 'validated', 'frames_dir'}
    for ts_name from the 'frame_lookup' section, or None if not registered.
    'frames' is {str(sec): {'z_value', 'sub_frame_path'}} for every SEC
    (aligned + dark)."""
    return _load().get('frame_lookup', {}).get('per_ts', {}).get(ts_name)


def resolve_frame(ts_name: str, *, sec: int = None, z_value: int = None,
                  acq_order: int = None) -> Optional[dict]:
    """
    Given exactly one of sec/z_value/acq_order, resolve the rest of that
    frame's identifiers plus its filename and full path from the
    'frame_lookup' section (both captured at register_frame_lookup() time,
    not composed here -- see its docstring for why).

    Returns {'sec', 'z_value', 'acq_order', 'is_dark', 'sub_frame_path',
    'frame_path'}, or None if frame_lookup isn't registered for ts_name or
    the given id isn't found. sub_frame_path/frame_path are None (not a
    lookup failure) if mdoc_data wasn't registered yet when frame_lookup
    was built for this TS. frame_path is frames_dir joined with
    sub_frame_path's basename (never sub_frame_path's own directory --
    that's typically a stale acquisition-PC path, see check_frames_found's
    docstring in validate_mdoc.py); None if frames_dir wasn't recorded
    (mdoc validated before the frames_dir field existed -- re-run
    validate-mdoc to backfill).
    """
    lookup = get_frame_lookup(ts_name)
    if lookup is None:
        return None

    frames = lookup['frames']
    if sec is not None:
        found_sec = sec
        entry = frames.get(str(sec))
        if entry is None:
            return None
    elif z_value is not None or acq_order is not None:
        target_z = z_value if z_value is not None else acq_order - 1
        match = next(((s, e) for s, e in frames.items() if e['z_value'] == target_z), None)
        if match is None:
            return None
        found_sec, entry = int(match[0]), match[1]
    else:
        raise ValueError('resolve_frame: exactly one of sec/z_value/acq_order is required')

    frames_dir = lookup.get('frames_dir')
    sub_frame_path = entry['sub_frame_path']
    frame_path = None
    if frames_dir and sub_frame_path:
        # sub_frame_path's own directory is typically a stale acquisition-PC
        # path (Windows/UNC); only its filename is meaningful once combined
        # with frames_dir, which is where the movie actually lives.
        basename = re.split(r'[\\/]+', sub_frame_path.strip())[-1]
        frame_path = str(Path(frames_dir) / basename)

    return {
        'sec':             found_sec,
        'z_value':         entry['z_value'],
        'acq_order':       entry['z_value'] + 1,
        'is_dark':         found_sec in set(lookup.get('dark_secs', [])),
        'sub_frame_path':  sub_frame_path,
        'frame_path':      frame_path,
    }


def get_analysis_runs() -> list:
    """
    List of {'kind', 'output_dir', 'label', 'timestamp'} records for every
    gain_check/aretomo_analyse/pytom_match run this project has completed --
    see record_analysis_run(). Used by shared.landing_page.write_landing_page
    to build analysis_start.html. Empty list if none recorded yet.
    """
    return _load().get('analysis_runs') or []


def record_analysis_run(kind: str, output_dir, label: str = None) -> None:
    """
    Append a completed run's record for the project landing page to link
    to, or refresh its timestamp/label if (kind, output_dir) already has
    one. Unlike this file's other project.json sections (which are
    last-write-wins -- see project_json.update_section), analysis_runs is
    a real history across every run, since a project commonly has more
    than one analyse/pytom-match run (e.g. cmd0 once, cmd1 several times
    with different parameters, cmd2 several times) and the landing page
    needs to list all of them, not just the most recent.

    kind       : 'gain_check' | 'aretomo_analyse' | 'pytom_match'
    output_dir : path to that run's own report directory (str or Path;
                 stored as given -- callers pass whatever they'd want a
                 human to see/use, absolute or relative to cwd)
    label      : human-readable run name shown on the landing page;
                 defaults to output_dir's own directory name, except when
                 that name collides with another already-recorded run
                 that has a *different* output_dir (a common case: several
                 `analyse --output <run-N>/analyse` runs all sharing the
                 leaf name "analyse") -- then both the new and the earlier
                 colliding entries are qualified with their parent
                 directory name instead, so the landing page never
                 silently merges two distinct runs onto one tab.
    """
    output_dir = str(output_dir)
    runs = get_analysis_runs()
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')

    def _leaf(od):
        return Path(od).name or od

    def _qualified(od):
        p = Path(od)
        leaf = p.name or od
        return f'{p.parent.name}/{leaf}' if p.parent.name else leaf

    if label is None:
        leaf = _leaf(output_dir)
        colliding = [r for r in runs
                     if r['output_dir'] != output_dir and _leaf(r['output_dir']) == leaf]
        if colliding:
            label = _qualified(output_dir)
            for r in colliding:
                if r['label'] == leaf:
                    r['label'] = _qualified(r['output_dir'])
        else:
            label = leaf

    for r in runs:
        if r.get('kind') == kind and r.get('output_dir') == output_dir:
            r['timestamp'] = timestamp
            r['label'] = label
            break
    else:
        runs.append({'kind': kind, 'output_dir': output_dir,
                      'label': label, 'timestamp': timestamp})

    update_section('analysis_runs', runs)


def get_tool_path(name: str) -> Optional[str]:
    """
    Return the recorded binary/env-dir path for an external tool (e.g.
    'aretomo3', 'pytom', 'imod', 'gapstop'), or None if never recorded.
    See record_tool_path()/resolve_tool_path().
    """
    return _load().get('tool_paths', {}).get(name)


def record_tool_path(name: str, path) -> None:
    """
    Remember an external tool's binary/env directory so later commands in
    this project can auto-fill it instead of the user re-typing
    --pytom-dir/--imod-bin-dir/--aretomo3/etc. on every invocation. Called
    either when a command runs with that path explicitly given (see
    resolve_tool_path()), or via enrich --set-path-<tool>.
    """
    paths = dict(_load().get('tool_paths', {}))
    paths[name] = str(path)
    update_section('tool_paths', paths)


def resolve_tool_path(name: str, cli_value):
    """
    CLI flag wins and gets remembered for next time (record_tool_path);
    otherwise fall back to whatever was previously recorded for this
    project, or None if neither is set (caller applies its own hardcoded
    default in that case -- this function never invents one).
    """
    if cli_value is not None:
        record_tool_path(name, cli_value)
        return cli_value
    return get_tool_path(name)
