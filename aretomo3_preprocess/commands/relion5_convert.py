"""
relion5-convert — convert AreTomo3 output to a RELION 5 tilt-series project.

Reads the AreTomo3 output directory and produces:
  <output>/tilt_series_aligned.star   — global STAR (one row per tomogram)
  <output>/tilt_series/ts-XXX.star    — per-TS STAR (one row per included tilt,
                                         sorted by pre-exposure)
  <output>/tilts/                     — unstacked per-tilt MRC files (optional)

Supports both cmd=0 (motion-correction + alignment) output directories and cmd=2
(reconstruction-only) directories where .mrc stacks are symlinks back to the
cmd=0 dir.  The cmd=0 directory is auto-detected by resolving the first .mrc
symlink found; it can be overridden with --cmd0-dir.

Movie file paths and per-tilt defocus targets are read from aretomo3_project.json
(populated by validate-mdoc and rename-ts in the standard pipeline).  If the
project file is missing, --mdoc-dir must be given as a fallback.

Typical usage
-------------
  # Standard pipeline — project.json has everything, no --mdoc-dir needed
  aretomo3-preprocess relion5-convert \\
      --input run001-cmd0 \\
      --output relion5 \\
      --dose 4.16 \\
      --movie-frames 8

  # Convert only a subset, skip MRC unstacking
  aretomo3-preprocess relion5-convert \\
      --input run001-cmd0 \\
      --output relion5 \\
      --dose 4.16 \\
      --movie-frames 8 \\
      --include ts-001 ts-002 \\
      --no-unstack

  # Also unstack half-datasets (EVN / ODD)
  aretomo3-preprocess relion5-convert \\
      --input run001-cmd0 \\
      --output relion5 \\
      --dose 4.16 \\
      --movie-frames 8 \\
      --unstack-halves

  # Fallback when project.json is absent
  aretomo3-preprocess relion5-convert \\
      --input run001-cmd0 \\
      --mdoc-dir frames \\
      --output relion5 \\
      --dose 4.16 \\
      --movie-frames 8
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import mrcfile as _mrcfile
    _HAS_MRCFILE = True
except ImportError:
    _HAS_MRCFILE = False

try:
    import starfile as _starfile
    _HAS_STARFILE = True
except ImportError:
    _HAS_STARFILE = False

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from aretomo3_preprocess.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file, parse_mdoc_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cmd0_dir(input_dir: Path) -> Path:
    """
    Try to locate the cmd=0 directory by resolving a .mrc symlink in input_dir.
    If no symlink is found (i.e. input_dir IS the cmd=0 dir) returns input_dir.
    """
    for mrc in sorted(input_dir.glob('ts-*.mrc')):
        if mrc.is_symlink():
            return mrc.resolve().parent
    return input_dir


def _read_session_json(path: Path) -> dict:
    """Load AreTomo3_Session.json; return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _build_prior_dose_map(tlt_data: dict) -> dict:
    """
    Build acq_order → prior_dose map from parsed TLT.txt data.

    RELION pre-exposure = total dose accumulated BEFORE this tilt was acquired.
    Frame with acq_order=1 has prior_dose=0.
    """
    rows_by_acq = sorted(tlt_data.values(), key=lambda r: r['acq_order'])
    prior = {}
    cumulative = 0.0
    for row in rows_by_acq:
        prior[row['acq_order']] = round(cumulative, 4)
        cumulative += row['dose_e_per_A2']
    return prior  # acq_order → dose before this tilt


def _parse_xf_file(xf_path: Path, pixel_size: float) -> list:
    """
    Parse IMOD _st.xf transformation file.

    Returns a list (one entry per tilt, 0-indexed, tilt-sorted) of dicts:
      {'z_rot', 'x_shift_angst', 'y_shift_angst', 'x_tilt'}
    """
    results = []
    with open(xf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) != 6:
                continue
            A11, A12, A21, A22, DX, DY = vals
            z_rot = math.degrees(math.atan2(A12, A11))
            T = np.array([[A11, A12, DX],
                          [A21, A22, DY],
                          [0.0, 0.0, 1.0]])
            T_inv = np.linalg.inv(T)
            x_shift_angst = T_inv[0, 2] * pixel_size
            y_shift_angst = T_inv[1, 2] * pixel_size
            results.append({
                'z_rot':         z_rot,
                'x_shift_angst': x_shift_angst,
                'y_shift_angst': y_shift_angst,
                'x_tilt':        0.0,
            })
    return results


def _parse_imod_tlt(tlt_path: Path) -> list:
    """
    Parse IMOD _st.tlt refined tilt angles.
    Returns a list of floats (0-indexed, tilt-sorted order).
    """
    tilts = []
    with open(tlt_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                tilts.append(float(line))
    return tilts


def _load_mdoc_from_project(project: dict, ts_name: str) -> tuple:
    """
    Extract per-frame mdoc data for ts_name from aretomo3_project.json.

    Returns (frames_dict, frames_dir) where:
      frames_dict : {z_value (int): {'sub_frame_path', 'target_defocus', ...}}
      frames_dir  : Path to the frames directory (parent of the original mdoc),
                    or None if not determinable.

    Returns ({}, None) if the TS is not found.
    """
    # Build ts_name → original_mdoc_path from rename_ts lookup
    rename_ts = project.get('rename_ts', {})
    original_mdoc_path = None
    for grid in rename_ts.get('grids', {}).values():
        lookup = grid.get('lookup', {})
        key = f'{ts_name}.mdoc'
        if key in lookup:
            original_mdoc_path = Path(lookup[key])
            break

    if original_mdoc_path is None:
        return {}, None

    original_stem = original_mdoc_path.stem   # e.g. 'Position_1'
    frames_dir    = original_mdoc_path.parent

    per_ts = project.get('mdoc_data', {}).get('per_ts', {})
    ts_data = per_ts.get(original_stem)
    if ts_data is None:
        return {}, frames_dir

    # frames keyed by string z_value in JSON → convert to int
    raw_frames = ts_data.get('frames', {})
    frames = {int(k): v for k, v in raw_frames.items()}
    return frames, frames_dir


def _unstack_mrc(src: Path, out_dir: Path, index_to_stem: dict,
                 pixel_size: float, suffix: str = '.mrc',
                 dry_run: bool = False) -> dict:
    """
    Unstack src MRC (tilt-sorted stack) into individual per-tilt MRC files.

    index_to_stem: {0-indexed stack frame → output stem (no extension)}
    Returns dict: {0-indexed frame → output Path}
    """
    if not _HAS_MRCFILE:
        print('  WARNING: mrcfile not installed — skipping unstack')
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    if not src.exists():
        print(f'  WARNING: {src} not found — skipping unstack')
        return {}

    if dry_run:
        for idx, stem in index_to_stem.items():
            out_path = out_dir / f'{stem}{suffix}'
            print(f'  [dry-run] would write {out_path}')
            written[idx] = out_path
        return written

    with _mrcfile.open(src, permissive=True) as mrc:
        stack = mrc.data
        voxel = mrc.voxel_size

    for idx, stem in index_to_stem.items():
        out_path = out_dir / f'{stem}{suffix}'
        if out_path.exists():
            written[idx] = out_path
            continue
        with _mrcfile.new(str(out_path), overwrite=True) as out:
            out.set_data(np.array(stack[idx], dtype=stack.dtype))
            out.voxel_size = voxel
        written[idx] = out_path

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Per-tomogram processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_ts(ts_name: str, input_dir: Path, cmd0_dir: Path,
                output_dir: Path, session: dict,
                movie_frames: int, mtf: str, optics_group: str,
                no_unstack: bool, unstack_halves: bool, dry_run: bool,
                project: dict = None, mdoc_dir: Path = None) -> dict | None:
    """
    Process one tilt series.  Returns a dict of global-star fields, or None on
    fatal error.

    mdoc data is sourced from project.json when available; mdoc_dir is only
    used as a fallback if project.json is missing or the TS is not in it.
    """
    # ── locate AreTomo3 files ─────────────────────────────────────────────────
    aln_path   = input_dir  / f'{ts_name}.aln'
    ctf_path   = input_dir  / f'{ts_name}_CTF.txt'
    tlt_path   = cmd0_dir   / f'{ts_name}_TLT.txt'
    xf_path    = input_dir  / f'{ts_name}_Imod' / f'{ts_name}_st.xf'
    itlt_path  = input_dir  / f'{ts_name}_Imod' / f'{ts_name}_st.tlt'
    mrc_path   = input_dir  / f'{ts_name}.mrc'
    ctf_mrc    = input_dir  / f'{ts_name}_CTF.mrc'

    for path, label in [
        (aln_path,  '.aln'),
        (ctf_path,  '_CTF.txt'),
        (tlt_path,  '_TLT.txt'),
        (xf_path,   '_Imod/_st.xf'),
        (itlt_path, '_Imod/_st.tlt'),
    ]:
        if not path.exists():
            print(f'  ERROR: {ts_name}: {label} not found at {path}')
            return None

    # ── load mdoc data (project JSON first, then file fallback) ───────────────
    mdoc_frames = {}
    frames_dir  = mdoc_dir   # may be None

    if project:
        mdoc_frames, proj_frames_dir = _load_mdoc_from_project(project, ts_name)
        if mdoc_frames:
            frames_dir = proj_frames_dir
        elif mdoc_dir is None:
            print(f'  WARNING: {ts_name}: not found in project.json mdoc_data '
                  f'and no --mdoc-dir given — movie paths will be unknown')

    if not mdoc_frames and mdoc_dir is not None:
        mdoc_path = mdoc_dir / f'{ts_name}.mdoc'
        if mdoc_path.exists():
            mdoc_frames, _ = parse_mdoc_file(mdoc_path)
            frames_dir = mdoc_dir
        else:
            print(f'  WARNING: {ts_name}: {mdoc_path} not found — movie paths will be unknown')

    # ── parse AreTomo3 files ──────────────────────────────────────────────────
    aln_data  = parse_aln_file(aln_path)
    ctf_data  = parse_ctf_file(ctf_path)
    tlt_data  = parse_tlt_file(tlt_path)
    pix_size  = session.get('parameters', {}).get('PixSize', 1.0)
    xf_list   = _parse_xf_file(xf_path, pix_size)
    itlt_list = _parse_imod_tlt(itlt_path)

    if not tlt_data:
        print(f'  ERROR: {ts_name}: no rows in _TLT.txt')
        return None

    n_tilts = len(tlt_data)

    dark_frame_bs  = {df['frame_b'] for df in aln_data['dark_frames']}
    prior_dose_map = _build_prior_dose_map(tlt_data)

    thickness_px = aln_data.get('thickness') or 0
    thickness_nm = round(thickness_px * pix_size / 10.0, 2)

    tilt_axis = session.get('parameters', {}).get('TiltAxis', [0.0])
    if isinstance(tilt_axis, list):
        tilt_axis = tilt_axis[0]

    # ── build per-tilt rows ───────────────────────────────────────────────────
    rows = []
    index_to_stem = {}   # 0-indexed stack frame → output stem

    for tlt_row_idx in range(1, n_tilts + 1):
        tlt_entry = tlt_data[tlt_row_idx]
        acq_order = tlt_entry['acq_order']
        z_value   = acq_order - 1
        is_dark   = tlt_row_idx in dark_frame_bs

        mdoc_entry     = mdoc_frames.get(z_value, {})
        sub_frame_path = mdoc_entry.get('sub_frame_path')
        if sub_frame_path and frames_dir:
            movie_path = frames_dir / sub_frame_path
            stem       = Path(sub_frame_path).stem
        else:
            movie_path = Path('FileNotFound')
            stem       = f'{ts_name}_{tlt_row_idx - 1:03d}'

        ctf_entry = ctf_data.get(tlt_row_idx, {})
        xf_idx    = tlt_row_idx - 1
        xf_entry  = xf_list[xf_idx] if xf_idx < len(xf_list) else {}
        y_tilt    = itlt_list[xf_idx] if xf_idx < len(itlt_list) else tlt_entry['nominal_tilt']
        pre_exp   = prior_dose_map.get(acq_order, 0.0)

        tilts_dir = output_dir / 'tilts'
        mrc_out   = tilts_dir / f'{stem}.mrc'
        evn_out   = tilts_dir / f'{stem}_EVN.mrc'
        odd_out   = tilts_dir / f'{stem}_ODD.mrc'

        if not is_dark:
            index_to_stem[xf_idx] = stem

        nominal_tilt = tlt_entry['nominal_tilt']

        rows.append({
            '_tlt_row':     tlt_row_idx,
            '_acq_order':   acq_order,
            '_is_dark':     is_dark,
            '_pre_exposure': pre_exp,
            'rlnTomoName':                  ts_name,
            'rlnMicrographMovieName':       str(movie_path),
            'rlnTomoTiltMovieFrameCount':   movie_frames,
            'rlnTomoNominalStageTiltAngle': nominal_tilt,
            'rlnTomoNominalTiltAxisAngle':  tilt_axis,
            'rlnMicrographPreExposure':     pre_exp,
            'rlnTomoNominalDefocus':        mdoc_entry.get('target_defocus', 0.0) or 0.0,
            'rlnCtfPowerSpectrum':          'FileNotFound',
            'rlnMicrographNameEven':        str(evn_out),
            'rlnMicrographNameOdd':         str(odd_out),
            'rlnMicrographName':            str(mrc_out),
            'rlnMicrographMetadata':        'FileNotFound',
            'rlnAccumMotionTotal':          0.0,
            'rlnAccumMotionEarly':          0.0,
            'rlnAccumMotionLate':           0.0,
            'rlnCtfImage':                  str(ctf_mrc),
            'rlnDefocusU':                  ctf_entry.get('defocus1_A', 0.0),
            'rlnDefocusV':                  ctf_entry.get('defocus2_A', 0.0),
            'rlnCtfAstigmatism':            ctf_entry.get('astig_A', 0.0),
            'rlnDefocusAngle':              ctf_entry.get('astig_angle_deg', 0.0),
            'rlnCtfFigureOfMerit':          ctf_entry.get('cc', 0.0),
            'rlnCtfMaxResolution':          ctf_entry.get('fit_spacing_A', 0.0),
            'rlnCtfIceRingDensity':         0.0,
            'rlnTomoXTilt':                 xf_entry.get('x_tilt', 0.0),
            'rlnTomoYTilt':                 y_tilt,
            'rlnTomoZRot':                  xf_entry.get('z_rot', 0.0),
            'rlnTomoXShiftAngst':           xf_entry.get('x_shift_angst', 0.0),
            'rlnTomoYShiftAngst':           xf_entry.get('y_shift_angst', 0.0),
            'rlnCtfScalefactor':            math.cos(math.radians(nominal_tilt)),
            'rlnIncluded':                  not is_dark,
        })

    # ── unstack MRC ───────────────────────────────────────────────────────────
    if not no_unstack:
        if not _HAS_MRCFILE:
            print('  WARNING: mrcfile not installed — cannot unstack MRC files')
        else:
            _unstack_mrc(mrc_path, output_dir / 'tilts', index_to_stem,
                         pixel_size=pix_size, suffix='.mrc', dry_run=dry_run)
            if unstack_halves:
                for sfx, src in [
                    ('_EVN.mrc', input_dir / f'{ts_name}_EVN.mrc'),
                    ('_ODD.mrc', input_dir / f'{ts_name}_ODD.mrc'),
                ]:
                    _unstack_mrc(src, output_dir / 'tilts', index_to_stem,
                                 pixel_size=pix_size, suffix=sfx, dry_run=dry_run)

    # ── write per-TS STAR ─────────────────────────────────────────────────────
    _TS_COLS = [
        'rlnTomoName', 'rlnMicrographMovieName', 'rlnTomoTiltMovieFrameCount',
        'rlnTomoNominalStageTiltAngle', 'rlnTomoNominalTiltAxisAngle',
        'rlnMicrographPreExposure', 'rlnTomoNominalDefocus',
        'rlnCtfPowerSpectrum', 'rlnMicrographNameEven', 'rlnMicrographNameOdd',
        'rlnMicrographName', 'rlnMicrographMetadata',
        'rlnAccumMotionTotal', 'rlnAccumMotionEarly', 'rlnAccumMotionLate',
        'rlnCtfImage',
        'rlnDefocusU', 'rlnDefocusV', 'rlnCtfAstigmatism', 'rlnDefocusAngle',
        'rlnCtfFigureOfMerit', 'rlnCtfMaxResolution', 'rlnCtfIceRingDensity',
        'rlnTomoXTilt', 'rlnTomoYTilt', 'rlnTomoZRot',
        'rlnTomoXShiftAngst', 'rlnTomoYShiftAngst',
        'rlnCtfScalefactor',
    ]

    ts_star_dir  = output_dir / 'tilt_series'
    ts_star_dir.mkdir(parents=True, exist_ok=True)
    ts_star_path = ts_star_dir / f'{ts_name}.star'

    included = [r for r in rows if not r['_is_dark']]
    included.sort(key=lambda r: r['_pre_exposure'])

    import pandas as pd
    df_ts = pd.DataFrame([{k: r[k] for k in _TS_COLS} for r in included])

    if not dry_run:
        _starfile.write({ts_name: df_ts}, ts_star_path)

    n_inc  = len(included)
    n_dark = sum(1 for r in rows if r['_is_dark'])
    print(f'  {ts_name}: {n_inc} tilts ({n_dark} dark excluded) → {ts_star_path}')

    ctf_hand_val = 1.0
    if ctf_data:
        ctf_hand_val = float(next(iter(ctf_data.values())).get('dfhand', 1.0))

    return {
        'rlnTomoName':                   ts_name,
        'rlnTomoTiltSeriesStarFile':      str(ts_star_path),
        'rlnVoltage':                     session.get('parameters', {}).get('kV', 300.0),
        'rlnSphericalAberration':         session.get('parameters', {}).get('Cs', 2.7),
        'rlnAmplitudeContrast':           session.get('parameters', {}).get('AmpContrast', 0.1),
        'rlnMicrographOriginalPixelSize': pix_size,
        'rlnTomoHand':                    ctf_hand_val,
        'rlnMtfFileName':                 mtf,
        'rlnOpticsGroupName':             optics_group,
        'rlnTomoTiltSeriesPixelSize':     pix_size,
        'rlnTomoZRot':                    xf_list[0]['z_rot'] if xf_list else 0.0,
        'rlnTomoTomogramThickness':       thickness_nm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'relion5-convert',
        help='Convert AreTomo3 output to a RELION 5 tilt-series project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--input', required=True, metavar='DIR',
                   help='AreTomo3 output directory (cmd=0 or cmd=2).')
    p.add_argument('--cmd0-dir', default=None, metavar='DIR',
                   help='cmd=0 directory containing _TLT.txt and Session.json.  '
                        'Auto-detected from .mrc symlinks if not given.')
    p.add_argument('--mdoc-dir', default=None, metavar='DIR',
                   help='Fallback directory containing ts-XXX.mdoc files.  '
                        'Not needed when aretomo3_project.json is present '
                        '(populated by the standard pipeline).')
    p.add_argument('--output', required=True, metavar='DIR',
                   help='Output directory for RELION 5 project files.')
    p.add_argument('--dose', required=True, type=float, metavar='FLOAT',
                   help='Electron dose per tilt (e⁻/Å²).')
    p.add_argument('--movie-frames', required=True, type=int, metavar='INT',
                   help='Number of movie frames per tilt (varies by dataset).')
    p.add_argument('--mtf', default='DUMMY', metavar='FILE',
                   help='MTF file name for the detector (default: DUMMY).')
    p.add_argument('--optics-group', default='optics1', metavar='NAME',
                   help='Optics group name (default: optics1).')
    p.add_argument('--include', nargs='+', default=None, metavar='TS',
                   help='Process only these tilt-series names (e.g. ts-001 ts-002).')
    p.add_argument('--exclude', nargs='+', default=None, metavar='TS',
                   help='Skip these tilt-series names.')
    p.add_argument('--no-unstack', action='store_true',
                   help='Do not unstack .mrc tilt-series stacks to per-tilt files.')
    p.add_argument('--unstack-halves', action='store_true',
                   help='Also unstack _EVN.mrc and _ODD.mrc half-dataset stacks.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be done without writing any files.')
    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── dependency checks ─────────────────────────────────────────────────────
    missing = []
    if not _HAS_STARFILE:
        missing.append('starfile')
    if not _HAS_PANDAS:
        missing.append('pandas')
    if missing:
        print(f'ERROR: missing packages: {", ".join(missing)}')
        print('       Install with: pip install ' + ' '.join(missing))
        sys.exit(1)

    # ── resolve paths ─────────────────────────────────────────────────────────
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    mdoc_dir   = Path(args.mdoc_dir).resolve() if args.mdoc_dir else None

    if not input_dir.is_dir():
        print(f'ERROR: --input {input_dir} not found')
        sys.exit(1)
    if mdoc_dir is not None and not mdoc_dir.is_dir():
        print(f'ERROR: --mdoc-dir {mdoc_dir} not found')
        sys.exit(1)

    cmd0_dir = Path(args.cmd0_dir).resolve() if args.cmd0_dir else _detect_cmd0_dir(input_dir)
    if cmd0_dir != input_dir:
        print(f'cmd0 directory: {cmd0_dir}')

    # ── load project JSON ─────────────────────────────────────────────────────
    from aretomo3_preprocess.shared.project_json import load as _load_project
    project = _load_project()
    has_mdoc_data = bool(project.get('mdoc_data', {}).get('per_ts'))
    if has_mdoc_data:
        print('Using aretomo3_project.json for movie paths and defocus targets.')
    elif mdoc_dir is None:
        print('WARNING: aretomo3_project.json has no mdoc_data and --mdoc-dir not given.')
        print('         Movie paths will be set to FileNotFound.')

    # ── load session JSON ─────────────────────────────────────────────────────
    session = _read_session_json(cmd0_dir / 'AreTomo3_Session.json')
    if not session:
        print(f'WARNING: AreTomo3_Session.json not found in {cmd0_dir}')
        print(f'         Voltage/Cs/AmpContrast/PixSize will be 0 — use --cmd0-dir to fix.')

    # ── discover TS names ─────────────────────────────────────────────────────
    aln_files = sorted(input_dir.glob('ts-*.aln'))
    if not aln_files:
        aln_files = sorted(input_dir.glob('*.aln'))
    if not aln_files:
        print(f'ERROR: no .aln files found in {input_dir}')
        sys.exit(1)

    all_ts = [f.stem for f in aln_files]

    if args.include:
        ts_list = [t for t in all_ts if t in set(args.include)]
        missing_ts = set(args.include) - set(all_ts)
        if missing_ts:
            print(f'WARNING: --include names not found: {sorted(missing_ts)}')
    else:
        ts_list = all_ts

    if args.exclude:
        excl = set(args.exclude)
        ts_list = [t for t in ts_list if t not in excl]

    if not ts_list:
        print('ERROR: no tilt series to process')
        sys.exit(1)

    print(f'Input:       {input_dir}')
    print(f'Output:      {output_dir}')
    print(f'Tilt series: {len(ts_list)} / {len(all_ts)} total')
    if not args.no_unstack:
        print(f'MRC unstack: yes'
              + (' (+ EVN/ODD)' if args.unstack_halves else ' (.mrc only)'))
    else:
        print('MRC unstack: skipped (--no-unstack)')
    if args.dry_run:
        print('[DRY RUN — no files will be written]')

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── process each TS ───────────────────────────────────────────────────────
    global_rows = []
    n_ok = n_fail = 0

    for ts_name in ts_list:
        print(f'\n{ts_name}:')
        result = _process_ts(
            ts_name        = ts_name,
            input_dir      = input_dir,
            cmd0_dir       = cmd0_dir,
            output_dir     = output_dir,
            session        = session,
            movie_frames   = args.movie_frames,
            mtf            = args.mtf,
            optics_group   = args.optics_group,
            no_unstack     = args.no_unstack,
            unstack_halves = args.unstack_halves,
            dry_run        = args.dry_run,
            project        = project,
            mdoc_dir       = mdoc_dir,
        )
        if result is None:
            n_fail += 1
        else:
            global_rows.append(result)
            n_ok += 1

    # ── write global STAR ─────────────────────────────────────────────────────
    if global_rows:
        import pandas as pd
        _GLOBAL_COLS = [
            'rlnTomoName', 'rlnTomoTiltSeriesStarFile',
            'rlnVoltage', 'rlnSphericalAberration', 'rlnAmplitudeContrast',
            'rlnMicrographOriginalPixelSize', 'rlnTomoHand',
            'rlnMtfFileName', 'rlnOpticsGroupName',
            'rlnTomoTiltSeriesPixelSize', 'rlnTomoZRot',
            'rlnTomoTomogramThickness',
        ]
        df_global = pd.DataFrame(global_rows)[_GLOBAL_COLS]
        global_star = output_dir / 'tilt_series_aligned.star'
        if not args.dry_run:
            _starfile.write({'global': df_global}, global_star)
        print(f'\nGlobal STAR: {global_star}  ({len(global_rows)} TS)')

    print(f'\nDone: {n_ok} OK, {n_fail} failed')
    if n_fail:
        sys.exit(1)
