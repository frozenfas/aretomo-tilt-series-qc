"""
run-aretomo3 — wrapper around the AreTomo3 binary for batch tilt-series processing.

Builds the AreTomo3 command from Python CLI arguments, prints it as a formatted
multi-line shell snippet with inline annotations, then runs it with live
stdout/stderr streaming captured to a log file.

Pipeline modes (--cmd)
----------------------
Each mode expects different inputs and handles stacks / .aln files differently:

  cmd  --in-prefix          --in-suffix  --in-skips               Input stacks           .aln files
  ---  -------------------  -----------  -----------------------  ---------------------  ------------------
   0   frames/ts-           mdoc         (none — mdoc mode)       raw movies via mdoc    created in --output
   1   <cmd0 output>/ts-    mrc          _CTF,_Vol,_EVN,_ODD      already in --in-prefix created in --output
   2   <prev run>/ts-       mrc          _CTF,_Vol,_EVN,_ODD      symlinked into         pre-exist in
                                                                   --output/              --in-prefix dir

For cmd=2, AreTomo3 reads the existing .aln from the --in-prefix directory and
writes new volumes (at the requested --at-bin / --split-sum) to --output.
Symlinks to the .aln files (from --in-prefix) and .mrc/.TLT stacks (from
project.json or --mrcdir) are created directly in --output/.  AreTomo3 is
pointed at --output/ so the source run is never modified.

TiltAxis and AlignZ (cmd=1/2)
------------------------------
Supply these in one of three ways (listed in decreasing precedence):
  1. Explicit:   --tilt-axis 86.5 3  --align-z 590
  2. Analysis:   --analysis run001_analysis  (reads global_suggested)
  3. Auto:       omit both — AreTomo3 estimates them

Before running, validates that:
  - the gain reference file exists (cmd=0 only)
  - input files are found matching the prefix/suffix/skips pattern
  - gain orientation matches the recommendation in aretomo3_project.json
    from a prior check-gain-transform run
  - pixel spacing (--apix) matches PixelSpacing in the mdoc files
  - voltage (--kv) matches Voltage in the mdoc files (if present)

Parameter mismatches are reported as warnings that block execution unless
--force is given.  Missing files are always fatal.

On success, saves the invocation to aretomo3_project.json.  After a cmd=0
run the output stacks are also registered under input_stacks so that
subsequent cmd=2 runs can locate them without --mrcdir.

Typical usage
-------------
  # cmd=0 — full pipeline (motion correction + alignment + reconstruction)
  aretomo3-preprocess run-aretomo3 \\
      --output run001 --gain estimated_gain.mrc \\
      --apix 1.63 --fm-dose 0.52 --flip-gain 1 \\
      --gpu 0 1 2 3 --dry-run

  # cmd=1 — re-align from existing stacks using global analysis values
  aretomo3-preprocess run-aretomo3 \\
      --in-prefix run001/ts- --in-suffix mrc \\
      --output run003 --analysis run001_analysis \\
      --cmd 1 --apix 1.63 --gpu 0 1 2 3 --dry-run

  # cmd=2 — reconstruct only (reuse run003 .aln, bin4+bin8, with EVN/ODD)
  aretomo3-preprocess run-aretomo3 \\
      --in-prefix run003/ts- --in-suffix mrc \\
      --output run004 --mrcdir run001 \\
      --cmd 2 --at-bin 4 8 --split-sum 1 \\
      --apix 1.63 --gpu 0 1 2 3 --dry-run
"""

import re
import sys
import json
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import (
    load_or_create, update_section, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import (
    get_angpix, register_input_stacks, resolve_selected_ts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Brief annotations for each AreTomo3 flag (shown in dry-run output)
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_COMMENTS = {
    '-InPrefix':     'input file prefix for batch discovery',
    '-InSuffix':     'input file suffix (mdoc=live pipeline, mrc=offline)',
    '-InSkips':      'filename patterns to skip',
    '-OutDir':       'output directory',
    '-LogDir':       'directory for AreTomo3 internal log files',
    '-Gain':         'gain reference file (.mrc or .gain)',
    '-FlipGain':     'flip gain: 0=none 1=flipud',
    '-RotGain':      'rotate gain: 0=none 1=90CCW 2=180 3=270CCW',
    '-Gpu':          'GPU IDs to use',
    '-Cmd':          'pipeline mode: 0=full 1=from-alignment 2=recon-only 3=CTF-only',
    '-Serial':       'seconds to wait for next series; 1=offline batch',
    '-PixSize':      'movie pixel size in Å/px',
    '-Kv':           'accelerating voltage in kV',
    '-Cs':           'spherical aberration in mm',
    '-AmpContrast':  'amplitude contrast ratio',
    '-FmDose':       'dose per raw frame in e⁻/Å²',
    '-McBin':        'motion-correction binning factor',
    '-McPatch':      'patch grid for local motion correction (X Y)',
    '-FmInt':        'frames per rendered frame: 1=TIFF, ~15=EER',
    '-EerSampling':  'EER super-resolution sampling factor',
    '-SplitSum':     'write odd/even half-sets for dose weighting',
    '-VolZ':         'reconstruction Z thickness in pixels (0=auto-estimate)',
    '-AtBin':        'tomogram binning factor(s); up to 3 for multi-resolution',
    '-AtPatch':      'local alignment patches (0 0=global only)',
    '-Wbp':          'reconstruction method: 1=WBP 0=SART',
    '-FlipVol':      'flip tomogram for conventional orientation',
    '-TiltCor':      'apply tilt angle offset correction',
    '-DarkTol':      'dark frame rejection tolerance',
    '-TiltAxis':     'initial tilt axis angle [refinement_flag]',
    '-AlignZ':       'slab thickness for alignment in pixels (auto if omitted)',
    '-Group':        'frame group sizes for global and local motion',
    '-CorrCTF':      'local CTF correction before reconstruction',
    '-OutXF':        'write IMOD XF transform files',
    '-OutImod':      'write IMOD support files for RELION',
}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num(x) -> str:
    """Format number, stripping trailing .0 from whole-number floats."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def _is_flag(token: str) -> bool:
    """True if token is an AreTomo3 flag (e.g. -InPrefix), not a negative number."""
    return len(token) >= 2 and token[0] == '-' and token[1].isalpha()


def _group_cmd(cmd: list) -> list:
    """Group a flat command list into [[executable], [flag, val, ...], ...]."""
    groups = [[cmd[0]]]
    for token in cmd[1:]:
        if _is_flag(token):
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


def _fmt_command(cmd: list, annotate: bool = False) -> str:
    """Format command as a multi-line shell snippet.

    With annotate=True, adds a brief # comment after each flag line.
    Comments are display-only — strip them before running in bash directly.
    """
    groups = _group_cmd(cmd)
    n = len(groups)
    lines = []

    for i, g in enumerate(groups):
        is_last = (i == n - 1)
        text = ' '.join(g)
        cont = '' if is_last else ' \\'

        if i == 0:
            lines.append(f'{text}{cont}')
        else:
            if annotate:
                flag = g[0] if _is_flag(g[0]) else None
                comment = _FLAG_COMMENTS.get(flag, '') if flag else ''
                body = f'    {text}{cont}'
                if comment:
                    body = f'{body:<52}  # {comment}'
                lines.append(body)
            else:
                lines.append(f'    {text}{cont}')

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_cmd(args) -> list:
    """Assemble the AreTomo3 command as a flat list of string tokens."""
    cmd = [args.aretomo3_bin]

    cmd += ['-InPrefix', args.in_prefix]
    cmd += ['-InSuffix', args.in_suffix]
    # Only pass file-type skip patterns to AreTomo3 (e.g. _CTF, _Vol).
    # TS-name patterns (e.g. ts-008) are handled by not symlinking those files,
    # so passing them to -InSkips is unnecessary and overflows AreTomo3's buffer.
    skips = [s for s in (args.in_skips or []) if s and not s.startswith('ts-')]
    if skips:
        cmd += ['-InSkips', ','.join(skips)]

    cmd += ['-OutDir',  args.output]
    cmd += ['-LogDir',  args.output]
    if args.gain is not None:
        cmd += ['-Gain',     args.gain]
        cmd += ['-FlipGain', _num(args.flip_gain)]
        cmd += ['-RotGain',  _num(args.rot_gain)]

    cmd += ['-Gpu'] + [_num(g) for g in args.gpu]
    cmd += ['-Cmd',    _num(args.cmd)]
    cmd += ['-Serial', _num(args.serial)]

    cmd += ['-PixSize',     _num(args.apix)]
    cmd += ['-Kv',          _num(args.kv)]
    cmd += ['-Cs',          _num(args.cs)]
    cmd += ['-AmpContrast', _num(args.amp_contrast)]

    if args.fm_dose is not None:
        cmd += ['-FmDose', _num(args.fm_dose)]
    cmd += ['-McBin',   _num(args.mc_bin)]
    cmd += ['-McPatch'] + [_num(v) for v in args.mc_patch]
    cmd += ['-FmInt',   _num(args.fm_int)]
    if args.eer_sampling is not None:
        cmd += ['-EerSampling', _num(args.eer_sampling)]
    cmd += ['-SplitSum', _num(args.split_sum)]

    cmd += ['-VolZ',     _num(args.vol_z)]
    cmd += ['-AtBin']  + [_num(v) for v in args.at_bin]
    cmd += ['-AtPatch'] + [_num(v) for v in args.at_patch]
    cmd += ['-Wbp',      _num(args.wbp)]
    cmd += ['-FlipVol',  _num(args.flip_vol)]

    cmd += ['-TiltCor', _num(args.tilt_cor)]
    cmd += ['-DarkTol', _num(args.dark_tol)]
    if args.tilt_axis is not None:
        cmd += ['-TiltAxis'] + [_num(v) for v in args.tilt_axis]
    if args.align_z is not None:
        cmd += ['-AlignZ', _num(args.align_z)]
    if args.group is not None:
        cmd += ['-Group'] + [_num(v) for v in args.group]

    cmd += ['-CorrCTF', _num(args.corr_ctf)]
    cmd += ['-OutXF',   _num(args.out_xf)]
    cmd += ['-OutImod', _num(args.out_imod)]

    if getattr(args, 'resume', False):
        cmd += ['-Resume', '1']

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_mdoc_metadata(mdoc_path: Path) -> dict:
    """Extract PixelSpacing and Voltage from a mdoc file (first occurrence of each)."""
    fields = {}
    try:
        with open(mdoc_path) as fh:
            for line in fh:
                for key in ('PixelSpacing', 'Voltage'):
                    if key not in fields:
                        m = re.match(rf'\s*{key}\s*=\s*([0-9.]+)', line)
                        if m:
                            fields[key] = float(m.group(1))
                if len(fields) == 2:
                    break
    except Exception:
        pass
    return fields


def _find_input_files(in_prefix: str, in_suffix: str,
                      in_skips: list = None) -> tuple:
    """Return (in_dir, pattern, sorted file list).  in_dir is a Path."""
    p = Path(in_prefix)
    in_dir = p.parent
    stem   = p.name
    pattern = f'{stem}*.{in_suffix}'
    files = sorted(in_dir.glob(pattern)) if in_dir.is_dir() else []
    if in_skips:
        files = [f for f in files if not any(s in f.stem for s in in_skips)]
    return in_dir, pattern, files


def _setup_cmd2_staging(out_dir: Path, src_in_dir: Path,
                        mrcdir: Path = None, in_skips: list = None,
                        dry_run: bool = False, selected_ts: set = None) -> Path:
    """
    Populate the output directory with symlinks for cmd=2 runs.

    Creates symlinks directly in out_dir to:
      - ts-xxx.aln      →  src_in_dir/ts-xxx.aln         (alignment files)
      - ts-xxx.mrc      →  <source>/ts-xxx.mrc            (tilt-series stacks)
      - ts-xxx_TLT.txt  →  <source>/ts-xxx_TLT.txt       (frame ordering)

    The MRC/TLT source is project.json input_stacks (from a prior cmd=0 run)
    or the explicit --mrcdir fallback.

    src_in_dir (e.g. run003) is NEVER modified — only read.

    Returns out_dir (the directory AreTomo3 should be pointed at).
    """
    sep = '═' * 70
    staging_dir = out_dir

    # ── Source for MRC stacks and _TLT.txt ────────────────────────────────
    proj   = load_or_create()
    stored = proj.get('input_stacks', {})
    stacks = stored.get('stacks', {})

    if stacks:
        source_label = f'project.json  (cmd=0 run {stored.get("timestamp", "?")})'
        src_root     = Path(stored.get('cmd0_outdir', '.'))
        all_ts       = sorted(stacks)
        mrc_sources  = {ts: Path(stacks[ts]['path']) for ts in all_ts}

    elif mrcdir is not None:
        if not mrcdir.is_dir():
            print(f'ERROR: --mrcdir {mrcdir} not found')
            sys.exit(1)
        skips        = [s for s in (in_skips or []) if s]
        src_root     = mrcdir.resolve()
        source_label = f'--mrcdir  {src_root}'
        src_files    = [f for f in sorted(mrcdir.glob('ts-*.mrc'))
                        if not any(s in f.stem for s in skips)]
        all_ts       = [f.stem for f in src_files]
        mrc_sources  = {f.stem: f.resolve() for f in src_files}

    else:
        # No MRC source — still create ALN symlinks (mrc validation will handle error)
        source_label = 'none'
        src_root     = None
        all_ts       = sorted(p.stem for p in src_in_dir.glob('ts-*.aln'))
        mrc_sources  = {}

    # ── Apply TS selection filter ──────────────────────────────────────────
    if selected_ts is not None:
        orig_n      = len(all_ts)
        all_ts      = [ts for ts in all_ts if ts in selected_ts]
        mrc_sources = {ts: mrc_sources[ts] for ts in all_ts if ts in mrc_sources}
        n_excl      = orig_n - len(all_ts)
        if n_excl:
            print(f'  TS selection: {n_excl} excluded, {len(all_ts)} remaining')

    staging_dir.mkdir(parents=True, exist_ok=True)

    # ── Build lists of links to create ────────────────────────────────────
    to_link_aln = []
    for ts_name in all_ts:
        aln_src = src_in_dir / f'{ts_name}.aln'
        aln_dst = staging_dir / f'{ts_name}.aln'
        if aln_src.exists() and not aln_dst.exists() and not aln_dst.is_symlink():
            to_link_aln.append((ts_name, aln_src.resolve()))

    to_link_mrc = []
    for ts_name, src in mrc_sources.items():
        mrc_dst = staging_dir / f'{ts_name}.mrc'
        if not mrc_dst.exists() and not mrc_dst.is_symlink():
            to_link_mrc.append((ts_name, Path(src)))

    to_link_tlt = []
    if src_root is not None:
        for ts_name in all_ts:
            tlt_src = src_root / f'{ts_name}_TLT.txt'
            tlt_dst = staging_dir / f'{ts_name}_TLT.txt'
            if tlt_src.exists() and not tlt_dst.exists() and not tlt_dst.is_symlink():
                to_link_tlt.append((ts_name, tlt_src.resolve()))

    if not to_link_aln and not to_link_mrc and not to_link_tlt:
        print(f'  Output dir {staging_dir}/ already populated — skipping symlink creation.')
        return staging_dir

    print(sep)
    print(f'  CMD=2 OUTPUT DIRECTORY  →  {staging_dir}/')
    print(f'  .aln source    : {src_in_dir}/')
    if src_root is not None:
        print(f'  MRC/TLT source : {source_label}')
    parts = []
    if to_link_aln: parts.append(f'{len(to_link_aln)} .aln')
    if to_link_mrc: parts.append(f'{len(to_link_mrc)} .mrc')
    if to_link_tlt: parts.append(f'{len(to_link_tlt)} _TLT.txt')
    print(f'  Linking        : {" + ".join(parts)}')
    if to_link_aln:
        print(f'    {to_link_aln[0][0]}.aln  →  {to_link_aln[0][1]}')
        if len(to_link_aln) > 1:
            print(f'    ... ({len(to_link_aln) - 1} more .aln)')
    if to_link_mrc:
        print(f'    {to_link_mrc[0][0]}.mrc  →  {to_link_mrc[0][1]}')
        if len(to_link_mrc) > 1:
            print(f'    ... ({len(to_link_mrc) - 1} more .mrc)')
    if to_link_tlt:
        print(f'    {to_link_tlt[0][0]}_TLT.txt  →  {to_link_tlt[0][1]}')
        if len(to_link_tlt) > 1:
            print(f'    ... ({len(to_link_tlt) - 1} more _TLT.txt)')
    print(sep)
    print()

    for ts_name, src in to_link_aln:
        (staging_dir / f'{ts_name}.aln').symlink_to(src)
    for ts_name, src in to_link_mrc:
        (staging_dir / f'{ts_name}.mrc').symlink_to(src)
    for ts_name, src in to_link_tlt:
        (staging_dir / f'{ts_name}_TLT.txt').symlink_to(src)

    return staging_dir



def _filter_alns_for_overlap(staging_dir: Path, src_in_dir: Path,
                             analysis_dir: Path,
                             threshold: float, dry_run: bool = False) -> int:
    """
    Demote low-overlap frames to DarkFrame entries for cmd=2 runs.

    Reads each ts-*.aln from src_in_dir (original source, never modified).
    For frames with overlap_pct < threshold, removes the data row and inserts
    a DarkFrame header entry so AreTomo3 cmd=2 excludes them from reconstruction.

    The modified .aln is written into staging_dir, replacing the existing
    symlink with a real file.  src_in_dir is NEVER touched.

    Re-running with a different threshold always re-reads from src_in_dir, so
    the staging copy is always consistent with the current threshold.

    Returns the total number of frames demoted across all TS.
    """
    tag = '[DRY RUN] ' if dry_run else ''
    sep = '─' * 70

    aln_json = Path(analysis_dir) / 'alignment_data.json'
    if not aln_json.exists():
        print(f'ERROR: --analysis {analysis_dir}: alignment_data.json not found')
        sys.exit(1)

    with open(aln_json) as fh:
        aln_data = json.load(fh)

    # Read from original source — never touch src_in_dir
    aln_files = sorted(Path(src_in_dir).glob('ts-*.aln'))
    if not aln_files:
        print(f'Warning: no ts-*.aln files found in {src_in_dir} for overlap filtering')
        return 0

    print(f'{tag}Overlap filter: demoting frames with overlap_pct < {threshold}%')
    print(f'  .aln source    : {src_in_dir}/')
    print(f'  Writing to     : {staging_dir}/')
    print(f'  Analysis source: {analysis_dir}/alignment_data.json')
    print(sep)

    n_total  = 0
    n_ts_mod = 0
    for aln_file in aln_files:
        ts_name = aln_file.stem
        ts_data = aln_data.get(ts_name)
        if not ts_data:
            continue

        # Collect SECs to demote from the analysis frames list
        demote = {}  # sec → overlap_pct
        for f in ts_data.get('frames', []):
            ov = f.get('overlap_pct')
            if ov is not None and ov < threshold:
                demote[f['sec']] = ov

        if not demote:
            continue

        # Show what will be demoted
        ov_strs = ', '.join(f'{demote[s]:.0f}%' for s in sorted(demote))
        print(f'  {ts_name}: {len(demote)} frame(s) → DarkFrame  (overlap: {ov_strs})')

        n_total  += len(demote)
        n_ts_mod += 1

        if dry_run:
            continue

        # ── Read from original source ──────────────────────────────────────────
        raw_lines = aln_file.read_text().splitlines(keepends=True)

        # Scan data rows for the TILT values of demoted SECs
        tilt_of = {}   # sec → tilt
        for line in raw_lines:
            stripped = line.strip()
            if not stripped.startswith('#') and stripped:
                parts = stripped.split()
                if len(parts) == 10:
                    try:
                        sec = int(parts[0])
                        if sec in demote:
                            tilt_of[sec] = float(parts[9])
                    except ValueError:
                        pass

        # ── Build modified .aln ────────────────────────────────────────────────
        # Strategy:
        #   • All lines pass through unchanged EXCEPT:
        #     - Existing DarkFrame lines are collected and removed from header
        #     - Data rows for demoted SECs are dropped
        #     - ALL DarkFrame entries (original + new) are re-inserted just
        #       before '# AlphaOffset', sorted by frame_a ascending.
        #       AreTomo3 appears to require monotonically increasing frame_a.

        # Collect existing DarkFrame entries from the source .aln
        existing_dark = []  # list of (frame_a, line_str)
        for line in raw_lines:
            stripped = line.rstrip('\n').strip()
            if stripped.startswith('# DarkFrame'):
                parts = stripped.split()
                # format: # DarkFrame = frame_a frame_b tilt
                if len(parts) >= 6:
                    try:
                        frame_a = int(parts[3])
                        existing_dark.append((frame_a, line if line.endswith('\n') else line + '\n'))
                    except (ValueError, IndexError):
                        existing_dark.append((999999, line if line.endswith('\n') else line + '\n'))

        # Build new DarkFrame lines for demoted SECs
        new_dark = []
        for sec in demote:
            tilt = tilt_of.get(sec, 0.0)
            frame_a = sec - 1
            new_dark.append((frame_a, f'# DarkFrame = {frame_a:5d}{sec:5d} {tilt:8.2f}\n'))

        # Merge and sort all DarkFrame entries by frame_a ascending
        all_dark = sorted(existing_dark + new_dark, key=lambda x: x[0])
        all_dark_lines = [line for _, line in all_dark]

        new_lines = []
        inserted = False
        for line in raw_lines:
            stripped = line.rstrip('\n').strip()

            # Skip existing DarkFrame lines — they will be re-inserted sorted
            if stripped.startswith('# DarkFrame'):
                continue

            # Insert all sorted DarkFrame lines immediately before # AlphaOffset
            if not inserted and stripped.startswith('# AlphaOffset'):
                new_lines.extend(all_dark_lines)
                inserted = True

            # Drop demoted data rows
            if not stripped.startswith('#') and stripped:
                parts = stripped.split()
                if len(parts) == 10:
                    try:
                        if int(parts[0]) in demote:
                            continue
                    except ValueError:
                        pass

            new_lines.append(line if line.endswith('\n') else line + '\n')

        # Edge case: no AlphaOffset line found — append at end of header block
        if not inserted:
            last_hdr = max(
                (i for i, l in enumerate(new_lines) if l.strip().startswith('#')),
                default=-1,
            )
            insert_at = last_hdr + 1
            new_lines = new_lines[:insert_at] + all_dark_lines + new_lines[insert_at:]

        # ── Write to staging dir (replace symlink with real file) ─────────────
        staging_aln = staging_dir / f'{ts_name}.aln'
        if staging_aln.exists() or staging_aln.is_symlink():
            staging_aln.unlink()
        staging_aln.write_text(''.join(new_lines))
        print(f'    Written to staging: {staging_aln.name}')

    print(sep)
    if n_total == 0:
        print(f'  No frames below {threshold}% overlap — .aln files not modified.\n')
    else:
        print(f'{tag}{n_total} frame(s) demoted across {n_ts_mod} TS.\n')
        if not dry_run:
            print(f'  Source .aln files in {src_in_dir}/ were NOT modified.\n')

    return n_total


def _load_global_params(analysis_dir: Path) -> dict:
    """Load global_suggested TiltAxis and AlignZ from an analyse output dir.

    Returns {'rot_deg': float, 'align_z_px': int} or {} if not found.
    """
    proj_path = analysis_dir / 'aretomo3_project.json'
    if not proj_path.exists():
        return {}
    try:
        with open(proj_path) as fh:
            proj = json.load(fh)
    except Exception:
        return {}
    return proj.get('analyse', {}).get('global_suggested', {})


def _validate(args) -> tuple:
    """Run pre-flight checks.

    Returns (errors, warnings) where:
      errors   — fatal, always abort
      warnings — abort unless --force
    """
    errors   = []
    warnings = []

    # ── 1. Gain / fm-dose (only needed for cmd 0 = full pipeline) ─────────
    if args.gain is not None:
        if not Path(args.gain).exists():
            errors.append(f'Gain file not found: {args.gain!r}')
    elif args.cmd == 0:
        errors.append('--gain is required for --cmd 0 (motion correction mode)')

    if args.fm_dose is None and args.cmd == 0:
        errors.append('--fm-dose is required for --cmd 0 (motion correction mode)')

    # ── 2. Input files found ───────────────────────────────────────────────
    in_dir, pattern, mdoc_files = _find_input_files(
        args.in_prefix, args.in_suffix, args.in_skips)
    if not in_dir.is_dir():
        errors.append(f'Input directory not found: {in_dir}/')
    elif not mdoc_files:
        if args.in_suffix == 'mrc' and args.cmd == 2:
            # cmd=2 dry-run: symlinks not yet created — check project.json/mrcdir
            proj_stacks = load_or_create().get('input_stacks', {}).get('stacks', {})
            mrcdir_ok   = args.mrcdir is not None and Path(args.mrcdir).is_dir()
            if proj_stacks:
                print(f'  Input files     : {len(proj_stacks)} stacks in '
                      f'project.json (symlinks will be created before run)')
            elif mrcdir_ok:
                n = len([f for f in Path(args.mrcdir).glob('ts-*.mrc')
                         if not any(s in f.stem for s in (args.in_skips or []))])
                print(f'  Input files     : {n} stacks from --mrcdir '
                      f'(symlinks will be created before run)')
            else:
                errors.append(
                    f'No .mrc files found in {in_dir}/ and no stack source '
                    f'available.\n'
                    f'       Options:\n'
                    f'         1. Run --cmd 0 first (registers stacks in '
                    f'project.json automatically)\n'
                    f'         2. Add --mrcdir /path/to/stacks'
                )
        else:
            errors.append(
                f'No .{args.in_suffix} files found matching {in_dir}/{pattern}\n'
                f'       (run rename-ts first if using mdoc mode)'
            )
    else:
        print(f'  Input files     : {len(mdoc_files)} .{args.in_suffix} files '
              f'found ({in_dir}/{pattern})')

    # ── 3. .aln files present (cmd=2 only) ────────────────────────────────
    if args.cmd == 2 and in_dir.is_dir():
        aln_files = sorted(in_dir.glob('ts-*.aln'))
        if not aln_files:
            errors.append(
                f'No ts-*.aln files found in {in_dir}/\n'
                f'       --in-prefix for cmd=2 should point to the directory '
                f'of a previous alignment run (e.g. run003/ts-)'
            )
        else:
            print(f'  .aln files      : {len(aln_files)} found in {in_dir}/')

    # ── 4. Gain transform vs gain_check JSON (only if gain provided) ──────
    try:
        project = load_or_create()
        gc = project.get('gain_check', {})
    except SystemExit:
        gc = {}

    if args.gain is None:
        print('  Gain check      : skipped (no gain — cmd != 0)')
    elif gc:
        rec_rot  = gc.get('aretomo3_rot_gain')
        rec_flip = gc.get('aretomo3_flip_gain')
        rec_best = gc.get('best_transform', '?')

        if rec_rot is not None and rec_flip is not None:
            if rec_rot != args.rot_gain or rec_flip != args.flip_gain:
                warnings.append(
                    f'Gain transform mismatch:\n'
                    f'       gain_check recommends  -RotGain {rec_rot} -FlipGain {rec_flip}'
                    f'  (best_transform={rec_best!r})\n'
                    f'       you are passing        --rot-gain {args.rot_gain}'
                    f' --flip-gain {args.flip_gain}'
                )
            else:
                print(f'  Gain transform  : OK  '
                      f'(-RotGain {rec_rot} -FlipGain {rec_flip}, '
                      f'best_transform={rec_best!r})')
    else:
        print('  Gain check      : no gain_check in project JSON '
              '(run check-gain-transform to verify orientation)')

    # ── 4. Pixel spacing and voltage vs mdoc ──────────────────────────────
    if mdoc_files:
        meta = _read_mdoc_metadata(mdoc_files[0])
        ps = meta.get('PixelSpacing')
        if ps is not None:
            if abs(ps - args.apix) > 0.02:
                warnings.append(
                    f'Pixel spacing mismatch:\n'
                    f'       mdoc PixelSpacing = {ps} Å/px\n'
                    f'       --apix            = {args.apix} Å/px\n'
                    f'       (from {mdoc_files[0].name})'
                )
            else:
                print(f'  PixelSpacing    : OK  ({ps} Å/px matches --apix {args.apix})')

        volt = meta.get('Voltage')
        if volt is not None:
            if abs(volt - args.kv) > 1:
                warnings.append(
                    f'Voltage mismatch:\n'
                    f'       mdoc Voltage = {volt} kV\n'
                    f'       --kv         = {args.kv} kV'
                )
            else:
                print(f'  Voltage         : OK  ({volt} kV matches --kv {args.kv})')

    return errors, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'run-aretomo3',
        help='Build and run an AreTomo3 command for batch tilt-series processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--output', '-o', required=True,
                     help='Output directory for AreTomo3 results (-OutDir)')
    req.add_argument('--gain', '-g', default=None,
                     help='Gain reference file (.mrc or .gain) (-Gain). '
                          'Required for --cmd 0 (motion correction); '
                          'not needed for --cmd 1+ (alignment/recon only).')
    req.add_argument('--apix', '-a', type=float, default=None,
                     help='Pixel size of input movies in Å/px (-PixSize). '
                          'Auto-read from project.json (mdoc_data) if omitted.')
    req.add_argument('--fm-dose', '-d', type=float, default=None,
                     help='Electron dose per raw frame in e⁻/Å² (-FmDose). '
                          'Required for --cmd 0 (motion correction); '
                          'not needed for --cmd 1+ (alignment/recon only).')
    req.add_argument('--gpu', '-G', type=int, nargs='+', required=True,
                     metavar='ID',
                     help='GPU ID(s) to use (-Gpu)')

    inp = p.add_argument_group('input / discovery')
    inp.add_argument('--in-prefix', default='frames/ts-',
                     help='Input file prefix for batch discovery (-InPrefix)')
    inp.add_argument('--in-suffix', default='mdoc',
                     help='Input file suffix for batch discovery (-InSuffix)')
    inp.add_argument('--in-skips', nargs='*', metavar='PATTERN',
                     default=['_CTF', '_Vol', '_EVN', '_ODD'],
                     help='Filename stem substrings to exclude from input discovery '
                          '(-InSkips). Default excludes AreTomo3 side outputs. '
                          'Pass an empty string to disable.')
    inp.add_argument('--mrcdir', default=None,
                     help='Directory containing ts-xxx.mrc stacks; only needed '
                          'when stacks are not registered in project.json (e.g. '
                          'stacks produced outside this tool). Stacks are '
                          'symlinked into --in-prefix directory.')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='Path to ts_selection.csv from select-ts; only the '
                          'selected TS are staged for cmd=2 runs. '
                          'Auto-loaded from project.json if omitted.')
    inp.add_argument('--serial', type=int, default=1,
                     help='Seconds to wait for the next tilt series; '
                          '1=offline batch (default); 0=do not wait '
                          '(use when processing a single TS) (-Serial)')

    mic = p.add_argument_group('microscope / acquisition')
    mic.add_argument('--kv', type=float, default=300.0,
                     help='Accelerating voltage in kV (-Kv)')
    mic.add_argument('--cs', type=float, default=2.7,
                     help='Spherical aberration in mm (-Cs)')
    mic.add_argument('--amp-contrast', type=float, default=0.1,
                     help='Amplitude contrast ratio (-AmpContrast)')
    mic.add_argument('--flip-gain', type=int, default=0,
                     help='Gain flip: 0=none 1=flipud (-FlipGain)')
    mic.add_argument('--rot-gain', type=int, default=0,
                     help='Gain rotation: 0=none 1=90CCW 2=180 3=270CCW (-RotGain)')
    mic.add_argument('--cmd', type=int, default=0,
                     help='Pipeline mode: 0=full 1=from-alignment 2=recon-only '
                          '3=CTF-only 4=rotate-axis-180 (-Cmd)')

    mc = p.add_argument_group('motion correction')
    mc.add_argument('--mc-bin', type=int, default=1,
                    help='Binning factor for motion correction (-McBin)')
    mc.add_argument('--mc-patch', type=int, nargs=2, default=[1, 1],
                    metavar=('X', 'Y'),
                    help='Patch grid for local motion correction (-McPatch X Y)')
    mc.add_argument('--fm-int', type=int, default=1,
                    help='Raw frames per rendered frame: 1=TIFF, ~15=EER (-FmInt)')
    mc.add_argument('--eer-sampling', type=int, default=None,
                    help='EER sampling rate; omitted if not given (-EerSampling)')
    mc.add_argument('--split-sum', type=int, default=1,
                    help='Write odd/even half-sets for dose weighting (-SplitSum)')

    ali = p.add_argument_group('alignment and reconstruction')
    ali.add_argument('--vol-z', type=int, default=2046,
                     help='Reconstruction Z in pixels; 0=auto-estimate (-VolZ)')
    ali.add_argument('--at-bin', type=float, nargs='+', default=[4.0],
                     metavar='BIN',
                     help='Tomogram binning; up to 3 values for multi-resolution (-AtBin)')
    ali.add_argument('--at-patch', type=int, nargs=2, default=[0, 0],
                     metavar=('X', 'Y'),
                     help='Local alignment patch grid; 0 0=global only (-AtPatch X Y)')
    ali.add_argument('--wbp', type=int, default=1,
                     help='Reconstruction: 1=WBP 0=SART (-Wbp)')
    ali.add_argument('--flip-vol', type=int, default=1,
                     help='Flip reconstructed volume (-FlipVol)')
    ali.add_argument('--tilt-cor', type=int, default=1,
                     help='Apply tilt angle offset correction (-TiltCor)')
    ali.add_argument('--dark-tol', type=float, default=0.7,
                     help='Dark frame rejection tolerance (-DarkTol)')
    ali.add_argument('--corr-ctf', type=int, default=1,
                     help='Local CTF correction before reconstruction (-CorrCTF)')
    ali.add_argument('--out-xf', type=int, default=1,
                     help='Write IMOD XF transform files (-OutXF)')
    ali.add_argument('--out-imod', type=int, default=1,
                     help='Write IMOD support files for RELION (-OutImod)')
    ali.add_argument('--analysis', default=None,
                     help='analyse output directory; loads global suggested '
                          'TiltAxis and AlignZ from aretomo3_project.json. '
                          'Also used by --filter-overlap. '
                          'Explicit --tilt-axis / --align-z take precedence.')
    ali.add_argument('--filter-overlap', type=float, default=None,
                     metavar='PCT',
                     help='cmd=2 only: before reconstruction, demote frames '
                          'with overlap_pct < PCT to DarkFrame in each .aln '
                          'file (reads alignment_data.json from --analysis). '
                          'Modified .aln files are written to --output/; '
                          'the source run is never touched.')
    ali.add_argument('--tilt-axis', type=float, nargs='+', default=None,
                     metavar='ANGLE',
                     help='Tilt axis angle [REFINE_FLAG]; overrides --analysis; '
                          'auto-detected if neither given (-TiltAxis)')
    ali.add_argument('--align-z', type=int, default=None,
                     help='Sample thickness for alignment in pixels; overrides '
                          '--analysis; auto-estimated if neither given (-AlignZ)')
    ali.add_argument('--group', type=int, nargs=2, default=None,
                     metavar=('GLOBAL', 'LOCAL'),
                     help='Frame grouping GLOBAL LOCAL; AreTomo3 default if omitted (-Group)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--aretomo3', dest='aretomo3_bin', default='AreTomo3',
                     help='Path to or name of the AreTomo3 executable')
    ctl.add_argument('--resume', action='store_true',
                     help='Pass -Resume 1 to AreTomo3: skip TS that already '
                          'have output files in --output. Staging is preserved.')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print the annotated command without executing')
    ctl.add_argument('--force', action='store_true',
                     help='Run despite parameter mismatch warnings')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out_dir    = Path(args.output)
    src_in_dir = Path(args.in_prefix).parent  # original source (e.g. run003)
    staging_dir = None

    # ── Auto-fill from project.json if not given on CLI ────────────────────
    if args.apix is None:
        args.apix = get_angpix()
    if args.apix is None:
        print('ERROR: --apix not provided and no pixel size found in project.json')
        print('       Run validate-mdoc first, or supply --apix explicitly.')
        sys.exit(1)

    # ── Staging directory (cmd=2 only) ────────────────────────────────────
    # cmd=1: --in-prefix points at the cmd=0 output dir which has the stacks;
    #        no staging needed, fresh .aln is written to OutDir.
    # cmd=2: symlinks to the .aln files (from src_in_dir) and .mrc/.TLT stacks
    #        (from project.json or --mrcdir) are created directly in --output/.
    #        AreTomo3 is pointed at --output/.  src_in_dir (e.g. run003) is
    #        NEVER modified.
    if args.in_suffix == 'mrc' and args.cmd == 2:
        mrcdir      = Path(args.mrcdir) if args.mrcdir is not None else None
        selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
        staging_dir = _setup_cmd2_staging(
            out_dir     = out_dir,
            src_in_dir  = src_in_dir,
            mrcdir      = mrcdir,
            in_skips    = args.in_skips,
            dry_run     = args.dry_run,
            selected_ts = selected_ts,
        )
        stem = Path(args.in_prefix).name      # e.g. 'ts-'
        args.in_prefix = str(staging_dir / stem)

    in_dir = Path(args.in_prefix).parent  # staging_dir for cmd=2, src_in_dir otherwise

    # ── Pre-flight validation ──────────────────────────────────────────────
    print('Pre-flight checks:')
    errors, warnings = _validate(args)

    if errors:
        print()
        for msg in errors:
            print(f'  ERROR   : {msg}')
        print('\nAborting.')
        sys.exit(1)

    if warnings:
        print()
        for msg in warnings:
            for i, line in enumerate(msg.splitlines()):
                label = 'WARNING :' if i == 0 else '         '
                print(f'  {label} {line}')
        if args.dry_run or args.force:
            print(f'\n  {"(dry-run, continuing)" if args.dry_run else "(--force, continuing)"}')
        else:
            print('\n  Use --force to run anyway, or fix the parameters above.')
            sys.exit(1)

    print()

    # ── Load global TiltAxis / AlignZ from analysis (if requested) ─────────
    # cmd=2 is recon-only: alignment parameters are encoded in the .aln file,
    # so TiltAxis / AlignZ from analysis are not applicable.  --analysis is
    # still accepted (needed for --filter-overlap) but params are not loaded.
    if args.analysis is not None:
        if args.cmd == 2:
            print(f'Note: cmd=2 — TiltAxis/AlignZ not loaded from --analysis '
                  f'(already encoded in .aln files)\n')
        else:
            ana_dir = Path(args.analysis)
            gp = _load_global_params(ana_dir)
            if not gp:
                print(f'Warning: no global_suggested found in '
                      f'{ana_dir}/aretomo3_project.json — skipping\n')
            else:
                print(f'Global parameters from {ana_dir}/:')
                rot_deg    = gp.get('rot_deg')
                align_z_px = gp.get('align_z_px')
                if rot_deg is not None:
                    if args.tilt_axis is None:
                        args.tilt_axis = [rot_deg]
                        print(f'  TiltAxis : {rot_deg}°  (analysis global_suggested)')
                    else:
                        print(f'  TiltAxis : {args.tilt_axis}  (explicit --tilt-axis, '
                              f'analysis={rot_deg}° ignored)')
                if align_z_px is not None:
                    if args.align_z is None:
                        args.align_z = align_z_px
                        print(f'  AlignZ   : {align_z_px} px  (analysis global_suggested)')
                    else:
                        print(f'  AlignZ   : {args.align_z} px  (explicit --align-z, '
                              f'analysis={align_z_px} px ignored)')
                print()

    # ── Overlap-based .aln filtering (cmd=2 only) ──────────────────────────
    if args.filter_overlap is not None:
        if args.cmd != 2:
            print('Warning: --filter-overlap only applies to --cmd 2 — ignoring.\n')
        elif args.analysis is None:
            print('ERROR: --filter-overlap requires --analysis '
                  '(alignment_data.json source).')
            sys.exit(1)
        else:
            _filter_alns_for_overlap(
                staging_dir  = staging_dir,
                src_in_dir   = src_in_dir,
                analysis_dir = Path(args.analysis),
                threshold    = args.filter_overlap,
                dry_run      = args.dry_run,
            )

    # ── Build and print command ────────────────────────────────────────────
    cmd = _build_cmd(args)
    prefix = '[DRY RUN] ' if args.dry_run else ''
    print(f'{prefix}AreTomo3 command:\n')
    print(_fmt_command(cmd, annotate=True))
    print()

    if args.dry_run:
        return

    # ── Check binary ───────────────────────────────────────────────────────
    if shutil.which(args.aretomo3_bin) is None and not Path(args.aretomo3_bin).is_file():
        print(f'ERROR: AreTomo3 binary not found: {args.aretomo3_bin!r}')
        print('       Ensure it is on PATH or use --aretomo3 /path/to/AreTomo3')
        sys.exit(1)

    # ── Run AreTomo3 with live streaming + log capture ─────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'run_aretomo3.log'

    print(f'Output directory : {out_dir}')
    print(f'Log file         : {log_path}')
    print('Running AreTomo3 (streaming live)...\n')

    log_lines = []
    with open(log_path, 'w') as log_fh:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)
                log_lines.append(line)

    returncode = proc.returncode

    # ── Surface warnings/errors from log ──────────────────────────────────
    _WARN_KEYWORDS = ('warning', 'error', 'fail', 'abort', 'fatal', 'segfault')
    # AreTomo3 reports alignment residuals as "Iteration  N  Error: X.XX" — not an error
    _IGNORE_PATTERNS = ('iteration', )
    flagged = [
        l.rstrip() for l in log_lines
        if any(kw in l.lower() for kw in _WARN_KEYWORDS)
        and not any(ig in l.lower() for ig in _IGNORE_PATTERNS)
    ]
    if flagged:
        print(f'\n{len(flagged)} line(s) with potential issues '
              f'(full log: {log_path}):')
        for line in flagged[:20]:
            print(f'  {line}')
        if len(flagged) > 20:
            print(f'  ... ({len(flagged) - 20} more — see log)')

    print()
    if returncode != 0:
        print(f'ERROR: AreTomo3 exited with code {returncode}')
        print('Project JSON not updated (recording only successful runs).')
        sys.exit(returncode)

    print(f'AreTomo3 finished successfully.')

    # ── Register cmd=0 output stacks and TLT dir for later runs ──────────
    if args.cmd == 0:
        register_input_stacks(out_dir, in_skips=args.in_skips, tlt_dir=out_dir)

    # ── Save to project JSON ───────────────────────────────────────────────
    update_section(
        section='run_aretomo3',
        values={
            'command':    ' '.join(cmd),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'output_dir': str(out_dir.resolve()),
            'log':        str(log_path.resolve()),
            'returncode': returncode,
        },
        backup_dir=out_dir,
    )
