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
project.json input_stacks) are created directly in --output/.  AreTomo3 is
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
subsequent cmd=2 runs can locate them automatically.

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
      --output run004 \\
      --cmd 2 --at-bin 4 8 --split-sum 1 \\
      --apix 1.63 --gpu 0 1 2 3 --dry-run
"""

import re
import sys
import csv
import json
import shutil
import datetime
import statistics
import subprocess
import threading
from pathlib import Path
import argparse

from tqdm import tqdm

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
    '-Sart':         'SART iterations and projections per subset (default: 20 5)',
    '-FlipVol':      'flip tomogram for conventional orientation',
    '-TiltCor':      'apply tilt angle offset correction [optional fixed offset in degrees]',
    '-DarkTol':      'dark frame rejection tolerance',
    '-TiltAxis':     'initial tilt axis angle [refinement_flag]',
    '-AlignZ':       'slab thickness for alignment in pixels (auto if omitted)',
    '-Group':        'frame group sizes for global and local motion',
    '-CorrCTF':      'local CTF correction before reconstruction',
    '-OutXF':        'write IMOD XF transform files',
    '-OutImod':      'write IMOD support files for RELION',
}

# Very short (2-3 word) summaries for the grouped preflight parameter dump --
# distinct from _FLAG_COMMENTS above, which is more verbose and used for the
# annotated dry-run command listing.
_FLAG_SHORT = {
    '-InPrefix':     'input prefix',
    '-InSuffix':     'input suffix',
    '-InSkips':      'skip patterns',
    '-OutDir':       'output dir',
    '-Gpu':          'GPU IDs',
    '-Cmd':          'pipeline mode',
    '-Serial':       'batch mode',
    '-PixSize':      'pixel size',
    '-Kv':           'voltage',
    '-Cs':           'spherical aberration',
    '-AmpContrast':  'amplitude contrast',
    '-Gain':         'gain reference',
    '-FlipGain':     'gain flip',
    '-RotGain':      'gain rotation',
    '-FmDose':       'dose per frame',
    '-McBin':        'motion-corr binning',
    '-McPatch':      'motion-corr patches',
    '-FmInt':        'frames per rendered frame',
    '-EerSampling':  'EER supersampling',
    '-SplitSum':     'odd/even halves',
    '-VolZ':         'reconstruction thickness',
    '-AtBin':        'tomogram binning',
    '-AtPatch':      'local align patches',
    '-Wbp':          'reconstruction method',
    '-Sart':         'SART iterations',
    '-FlipVol':      'volume flip',
    '-TiltCor':      'tilt offset correction',
    '-DarkTol':      'dark-frame tolerance',
    '-TiltAxis':     'tilt axis angle',
    '-AlignZ':       'alignment thickness',
    '-Group':        'frame grouping',
    '-CorrCTF':      'CTF correction',
    '-OutXF':        'IMOD xf output',
    '-OutImod':      'IMOD support files',
}

# Grouping for the preflight parameter dump, mirroring AreTomo3's own log
# sections ("Motion correction input parameters" / "Tomography input
# parameters"), with a third "General" group for I/O/global flags that
# AreTomo3 prints ungrouped at the top of its own dump.
_FLAG_GROUP = {
    '-InPrefix': 'general', '-InSuffix': 'general', '-InSkips': 'general',
    '-OutDir': 'general', '-Gpu': 'general', '-Cmd': 'general',
    '-Serial': 'general', '-PixSize': 'general', '-Kv': 'general',
    '-Cs': 'general', '-AmpContrast': 'general',

    '-Gain': 'motion', '-FlipGain': 'motion', '-RotGain': 'motion',
    '-FmDose': 'motion', '-McBin': 'motion', '-McPatch': 'motion',
    '-FmInt': 'motion', '-EerSampling': 'motion', '-SplitSum': 'motion',

    '-VolZ': 'tomo', '-AtBin': 'tomo', '-AtPatch': 'tomo', '-Wbp': 'tomo',
    '-Sart': 'tomo', '-FlipVol': 'tomo', '-TiltCor': 'tomo',
    '-DarkTol': 'tomo', '-TiltAxis': 'tomo', '-AlignZ': 'tomo',
    '-Group': 'tomo', '-CorrCTF': 'tomo', '-OutXF': 'tomo', '-OutImod': 'tomo',
}
_GROUP_TITLES = {
    'general': 'General',
    'motion':  'Motion correction parameters',
    'tomo':    'Alignment / reconstruction parameters',
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


def _print_grouped_params(cmd: list) -> None:
    """
    Print the resolved AreTomo3 parameters grouped like AreTomo3's own log
    output (see e.g. 'Motion correction input parameters' / 'Tomography
    input parameters' sections in run_aretomo3.log), each with a short
    (2-3 word) summary of what it controls.
    """
    by_section = {'general': [], 'motion': [], 'tomo': []}
    for g in _group_cmd(cmd)[1:]:   # skip the executable itself
        flag = g[0]
        by_section.setdefault(_FLAG_GROUP.get(flag, 'general'), []).append(g)

    for key in ('general', 'motion', 'tomo'):
        entries = by_section.get(key, [])
        if not entries:
            continue
        title = _GROUP_TITLES[key]
        print(title)
        print('-' * len(title))
        for g in entries:
            flag, vals = g[0], ' '.join(g[1:])
            short = _FLAG_SHORT.get(flag, '')
            line = f'  {flag:<14} {vals}'
            if short:
                line = f'{line:<40}  ({short})'
            print(line)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────

def _effective_in_skips(in_skips: list) -> list:
    """
    Filter -InSkips patterns the same way _build_cmd does before handing
    them to AreTomo3: only file-type skip patterns (e.g. _CTF, _Vol) are
    passed to -InSkips. TS-name patterns (e.g. ts-008) are dropped here --
    passing them overflows AreTomo3's -InSkips buffer, and for cmd=2 staging
    they're meant to be handled by not symlinking those files instead. For
    cmd=0/1 (no staging step), a ts-name pattern currently has no exclusion
    mechanism at all and is silently ignored by AreTomo3.

    Preflight file/mdoc discovery must filter with this SAME effective list
    (not the raw args.in_skips) so it validates the file set AreTomo3 will
    actually see, not a smaller one that a ts-name pattern only appears to
    exclude.
    """
    return [s for s in (in_skips or []) if s and not s.startswith('ts-')]


def _build_cmd(args) -> list:
    """Assemble the AreTomo3 command as a flat list of string tokens."""
    cmd = [args.aretomo3_bin]

    cmd += ['-InPrefix', args.in_prefix]
    cmd += ['-InSuffix', args.in_suffix]
    skips = _effective_in_skips(args.in_skips)
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
    if args.sart is not None and args.wbp == 0:
        cmd += ['-Sart'] + [_num(v) for v in args.sart]
    cmd += ['-FlipVol',  _num(args.flip_vol)]

    tilt_cor_vals = args.tilt_cor
    if len(tilt_cor_vals) > 2:
        raise ValueError('--tilt-cor accepts at most two values: ENABLE [ANGLE]')
    cmd += ['-TiltCor'] + [_num(v) for v in tilt_cor_vals]
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
                        in_skips: list = None,
                        dry_run: bool = False, selected_ts: set = None,
                        split_sum: int = 0) -> Path:
    """
    Populate the output directory with symlinks for cmd=2 runs.

    Copies .aln files and creates symlinks in out_dir for all other files:
      - ts-xxx.aln      ←  src_in_dir/ts-xxx.aln         (copied — editable by overlap filter)
      - ts-xxx.mrc      →  <source>/ts-xxx.mrc            (tilt-series stacks)
      - ts-xxx_EVN.mrc  →  <source>/ts-xxx_EVN.mrc       (even frames, split_sum=1)
      - ts-xxx_ODD.mrc  →  <source>/ts-xxx_ODD.mrc       (odd frames, split_sum=1)
      - ts-xxx_TLT.txt  →  <source>/ts-xxx_TLT.txt       (frame ordering)

    The MRC/TLT source is project.json input_stacks (from a prior cmd=0 run).
    If not present, exits with an actionable error.

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

    else:
        print('ERROR: input_stacks not found in project.json.')
        print('       MRC stacks must be registered before running cmd=2.')
        print('       Register the cmd=0 output directory:')
        print('         aretomo3-preprocess enrich --mrc-data <run001/>')
        sys.exit(1)

    # ── Apply TS selection filter ──────────────────────────────────────────
    if selected_ts is not None:
        orig_n      = len(all_ts)
        all_ts      = [ts for ts in all_ts if ts in selected_ts]
        mrc_sources = {ts: mrc_sources[ts] for ts in all_ts if ts in mrc_sources}
        n_excl      = orig_n - len(all_ts)
        if n_excl:
            print(f'  TS selection: {n_excl} excluded, {len(all_ts)} remaining')

    # ── Drop TS with no .aln to reconstruct from ────────────────────────────
    # cmd=2 is recon-only: a TS staged without its .aln can never actually be
    # reconstructed (AreTomo3 reads the existing alignment; it doesn't
    # recompute one). Staging its .mrc anyway just leaves it sitting there
    # forever, and the post-run completeness check would flag it as
    # "incomplete" for a TS that was never eligible in the first place.
    orig_n      = len(all_ts)
    all_ts      = [ts for ts in all_ts if (src_in_dir / f'{ts}.aln').exists()]
    mrc_sources = {ts: mrc_sources[ts] for ts in all_ts}
    n_no_aln    = orig_n - len(all_ts)
    if n_no_aln:
        print(f'  No .aln in {src_in_dir}/: {n_no_aln} TS excluded, '
              f'{len(all_ts)} remaining')

    # ── Identify stale symlinks from a prior, broader staging pass ─────────
    # This function only ever ADDS symlinks; a previous run into this same
    # --output with a wider (or no) --select-ts can leave symlinks for TS
    # that are no longer in the current selection. AreTomo3 processes
    # whatever it finds via -InPrefix regardless of what this wrapper thinks
    # is selected, so those stale TS would otherwise still get reconstructed
    # -- and the post-run completeness check, which only checks the CURRENT
    # selection, would never notice either way.
    stale_ts = []
    if staging_dir.is_dir():
        staged_mrc = {
            p.stem for p in staging_dir.glob('ts-*.mrc')
            if not (p.stem.endswith('_EVN') or p.stem.endswith('_ODD'))
        }
        stale_ts = sorted(staged_mrc - set(all_ts))

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

    # _CTF.txt comes from the alignment run (src_in_dir), not from the cmd=0 root
    to_link_ctf = []
    for ts_name in all_ts:
        ctf_src = src_in_dir / f'{ts_name}_CTF.txt'
        ctf_dst = staging_dir / f'{ts_name}_CTF.txt'
        if ctf_src.exists() and not ctf_dst.exists() and not ctf_dst.is_symlink():
            to_link_ctf.append((ts_name, ctf_src.resolve()))

    to_link_evn = []
    to_link_odd = []
    if split_sum:
        for ts_name, src in mrc_sources.items():
            src_path = Path(src)
            evn_src  = src_path.parent / f'{ts_name}_EVN.mrc'
            odd_src  = src_path.parent / f'{ts_name}_ODD.mrc'
            evn_dst  = staging_dir / f'{ts_name}_EVN.mrc'
            odd_dst  = staging_dir / f'{ts_name}_ODD.mrc'
            if evn_src.exists() and not evn_dst.exists() and not evn_dst.is_symlink():
                to_link_evn.append((ts_name, evn_src.resolve()))
            if odd_src.exists() and not odd_dst.exists() and not odd_dst.is_symlink():
                to_link_odd.append((ts_name, odd_src.resolve()))

    nothing_to_do = not any([to_link_aln, to_link_mrc, to_link_tlt, to_link_ctf,
                              to_link_evn, to_link_odd, stale_ts])
    if nothing_to_do:
        print(f'  Output dir {staging_dir}/ already populated — skipping symlink creation.')
        return staging_dir

    prefix = '[DRY RUN] ' if dry_run else ''
    print(sep)
    print(f'  {prefix}CMD=2 OUTPUT DIRECTORY  →  {staging_dir}/')
    print(f'  {prefix}.aln source    : {src_in_dir}/')
    if src_root is not None:
        print(f'  {prefix}MRC/TLT source : {source_label}')
    copy_parts = []
    link_parts = []
    if to_link_aln: copy_parts.append(f'{len(to_link_aln)} .aln')
    if to_link_mrc: link_parts.append(f'{len(to_link_mrc)} .mrc')
    if to_link_evn: link_parts.append(f'{len(to_link_evn)} _EVN.mrc')
    if to_link_odd: link_parts.append(f'{len(to_link_odd)} _ODD.mrc')
    if to_link_tlt: link_parts.append(f'{len(to_link_tlt)} _TLT.txt')
    if to_link_ctf: link_parts.append(f'{len(to_link_ctf)} _CTF.txt')
    if copy_parts:
        print(f'  {prefix}Copying        : {" + ".join(copy_parts)}')
    if link_parts:
        print(f'  {prefix}Linking        : {" + ".join(link_parts)}')
    if to_link_aln:
        print(f'    {to_link_aln[0][0]}.aln  →  {to_link_aln[0][1]}')
        if len(to_link_aln) > 1:
            print(f'    ... ({len(to_link_aln) - 1} more .aln)')
    if to_link_mrc:
        print(f'    {to_link_mrc[0][0]}.mrc  →  {to_link_mrc[0][1]}')
        if len(to_link_mrc) > 1:
            print(f'    ... ({len(to_link_mrc) - 1} more .mrc)')
    if to_link_evn:
        print(f'    {to_link_evn[0][0]}_EVN.mrc  →  {to_link_evn[0][1]}')
        if len(to_link_evn) > 1:
            print(f'    ... ({len(to_link_evn) - 1} more _EVN.mrc)')
    if to_link_odd:
        print(f'    {to_link_odd[0][0]}_ODD.mrc  →  {to_link_odd[0][1]}')
        if len(to_link_odd) > 1:
            print(f'    ... ({len(to_link_odd) - 1} more _ODD.mrc)')
    if to_link_tlt:
        print(f'    {to_link_tlt[0][0]}_TLT.txt  →  {to_link_tlt[0][1]}')
        if len(to_link_tlt) > 1:
            print(f'    ... ({len(to_link_tlt) - 1} more _TLT.txt)')
    if to_link_ctf:
        print(f'    {to_link_ctf[0][0]}_CTF.txt  →  {to_link_ctf[0][1]}')
        if len(to_link_ctf) > 1:
            print(f'    ... ({len(to_link_ctf) - 1} more _CTF.txt)')
    if stale_ts:
        print(f'  {prefix}Removing stale : {len(stale_ts)} TS no longer selected')
        print(f'    {stale_ts[0]}')
        if len(stale_ts) > 1:
            print(f'    ... ({len(stale_ts) - 1} more)')
    print(sep)
    print()

    if dry_run:
        return staging_dir

    # Remove stale symlinks left by a prior, broader staging pass. Never
    # touch the copied .aln (user-editable, per this function's docstring)
    # -- with its .mrc symlink gone, AreTomo3 simply has nothing to
    # reconstruct for that TS, which is all that's needed.
    for ts_name in stale_ts:
        for suffix in ('.mrc', '_EVN.mrc', '_ODD.mrc', '_TLT.txt', '_CTF.txt'):
            p = staging_dir / f'{ts_name}{suffix}'
            if p.is_symlink():
                p.unlink()
        stale_aln = staging_dir / f'{ts_name}.aln'
        if stale_aln.exists():
            print(f'    NOTE: {stale_aln} left in place (copied file, not '
                  f'auto-deleted) — {ts_name} will not be reprocessed since '
                  f'its .mrc symlink was removed.')

    all_ops = (
        [(ts, src, f'{ts}.aln',     'copy') for ts, src in to_link_aln] +
        [(ts, src, f'{ts}.mrc',     'link') for ts, src in to_link_mrc] +
        [(ts, src, f'{ts}_EVN.mrc', 'link') for ts, src in to_link_evn] +
        [(ts, src, f'{ts}_ODD.mrc', 'link') for ts, src in to_link_odd] +
        [(ts, src, f'{ts}_TLT.txt', 'link') for ts, src in to_link_tlt] +
        [(ts, src, f'{ts}_CTF.txt', 'link') for ts, src in to_link_ctf]
    )

    staging_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(all_ops, desc='  Staging', unit='file', ncols=72) as pbar:
        for ts_name, src, fname, op in pbar:
            dst = staging_dir / fname
            if op == 'copy':
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)

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

        # ── Write to staging dir (overwrites the copied .aln) ─────────────────
        staging_aln = staging_dir / f'{ts_name}.aln'
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


def _check_mdocs(mdoc_files: list) -> tuple:
    """
    Validate mdoc parsing for every discovered file and collect basic
    per-TS stats.

    Reuses validate-mdoc's AreTomo3 parser simulation (missing fields AND
    out-of-order fields, e.g. the AreTomo3 >= 2.3.0 ExposureTime ordering
    issue) so a mdoc that would silently corrupt tilt indices is caught
    here as a fatal error, rather than AreTomo3 quietly recovering the
    wrong number of tilts partway through a batch run.

    Returns (errors, stats).  errors is a list of fatal messages, one per
    distinct failure reason (collapsed with a count + example filename,
    since a shared mdoc collection template typically means the same
    problem affects every TS, not just one); stats has n_mdocs / n_valid /
    median_tilts / min_tilts / max_tilts / median_frames_per_tilt / n_v222 /
    n_v230 / expected_sections (dict of ts_name -> n_sections, for the
    post-run tilt-count check) -- any numeric field may be None if no mdoc
    parsed successfully.

    Fatal/abort is always keyed to AreTomo3 >= 2.3.0 conformance (the
    stricter, currently-shipping ruleset), regardless of what's actually
    installed -- being conservative here is cheap insurance. v222/v230
    conformance is reported separately so a "worked before, breaks on
    2.3.0" mdoc (passes v222, fails v230) is visible even when the exact
    installed AreTomo3 version isn't known at preflight time.
    """
    from aretomo3_preprocess.commands.validate_mdoc import validate_file
    from aretomo3_preprocess.shared.parsers import parse_mdoc_file

    bad_by_reason = {}   # reason -> [filenames]
    tilts_per_ts = []
    frames_per_tilt = []
    expected_sections = {}   # ts_name -> n_sections (for post-run check)
    n_v222 = n_v230 = 0

    for path in mdoc_files:
        result = validate_file(path)
        expected_sections[path.stem] = result['n_sections']
        if result['passes_v222']:
            n_v222 += 1
        if result['passes_v230']:
            n_v230 += 1

        if not result['success']:
            reason = result['failure'] or 'field order'
            bad_by_reason.setdefault(reason, []).append(path.name)
            continue
        tilts_per_ts.append(result['n_tilts'])

        # validate_file()'s hand-rolled AreTomo3-simulation passed above, but
        # parse_mdoc_file() is a separate (mdocfile-library-based) parser
        # that can still fail on a quirk the simulation doesn't check for --
        # don't let that crash the whole preflight with an unhandled
        # traceback; surface it as the same kind of fatal error as any other
        # mdoc problem caught here.
        try:
            mdoc_frames, _ = parse_mdoc_file(path)
        except Exception as e:
            bad_by_reason.setdefault(f'mdocfile parse error: {e}', []).append(path.name)
            continue
        nsub = [f['num_subframes'] for f in mdoc_frames.values()
                if f.get('num_subframes')]
        if nsub:
            frames_per_tilt.append(statistics.median(nsub))

    # One collapsed error per distinct failure reason, not one per file --
    # with many mdocs sharing the same collection template, the same
    # problem (e.g. a field-order issue) typically affects all of them.
    errors = []
    in_dir = mdoc_files[0].parent
    for reason, names in bad_by_reason.items():
        example = names[0] if len(names) == 1 else f'{names[0]} (+{len(names) - 1} more)'
        errors.append(
            f'{len(names)}/{len(mdoc_files)} mdoc(s) would silently fail/corrupt '
            f'(reason={reason}): {example}\n'
            f'       run validate-mdoc for details/fix, e.g.:\n'
            f'         aretomo3-preprocess validate-mdoc {in_dir}/*.mdoc'
        )

    stats = {
        'n_mdocs':      len(mdoc_files),
        'n_valid':      len(tilts_per_ts),
        'n_v222':       n_v222,
        'n_v230':       n_v230,
        'median_tilts': int(statistics.median(tilts_per_ts)) if tilts_per_ts else None,
        'min_tilts':    min(tilts_per_ts) if tilts_per_ts else None,
        'max_tilts':    max(tilts_per_ts) if tilts_per_ts else None,
        'median_frames_per_tilt':
            int(statistics.median(frames_per_tilt)) if frames_per_tilt else None,
        'expected_sections': expected_sections,
    }
    return errors, stats


def _detect_aretomo3_version(binary: str) -> str:
    """
    Best-effort AreTomo3 version string for the preflight log.

    AreTomo3 has no confirmed stable --version flag across releases, so
    this tries a few common patterns with a short timeout and falls back
    to the resolved binary path + modification date (often reflects an
    install/build date, sometimes a version-named directory) if none
    produce a recognizable version string. Never authoritative -- just a
    transparency aid so the log always states what actually ran, given
    this whole preflight exists because of a version-specific parser
    change.
    """
    resolved = shutil.which(binary) or binary
    for flag in ('--version', '-version', '-v'):
        try:
            proc = subprocess.run([binary, flag], capture_output=True,
                                  text=True, timeout=5)
        except Exception:
            continue
        out = (proc.stdout + proc.stderr).strip()
        m = re.search(r'\d+\.\d+(\.\d+)?', out)
        if m:
            return f'{m.group(0)} (via {flag})'
    try:
        mtime = datetime.datetime.fromtimestamp(
            Path(resolved).stat().st_mtime).strftime('%Y-%m-%d')
        return f'unknown (no --version support found; binary: {resolved}, modified {mtime})'
    except OSError:
        return f'unknown (binary not found at {resolved!r})'


def _check_gpus(requested_gpus: list) -> list:
    """
    Best-effort GPU availability check via nvidia-smi.

    Returns a list of warning strings. Empty if nvidia-smi isn't available
    (not itself a problem -- e.g. it may not be on PATH even though CUDA
    works) or every requested GPU ID is present.
    """
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    available = {int(x) for x in proc.stdout.split() if x.strip().isdigit()}
    if not available:
        return []
    missing = [g for g in requested_gpus if g not in available]
    if missing:
        return [f'Requested GPU ID(s) not found by nvidia-smi: {missing} '
                f'(available: {sorted(available)})']
    return []


def _check_disk_space(out_dir: Path) -> list:
    """
    Warn if free space on --output's filesystem is low.

    A precise "enough space for this run" estimate depends on VolZ/AtBin/
    dtype/split-sum and is easy to get wrong; this only flags an absolute
    low-water mark as a sanity check, not a prediction. (A batch tomogram
    run genuinely crashed the project.json write from exactly this cause
    -- see project_json.py's atomic-write fix -- so even a rough warning
    here is worth having.)
    """
    check_dir = out_dir if out_dir.exists() else out_dir.parent
    try:
        usage = shutil.disk_usage(check_dir)
    except OSError:
        return []
    free_gb = usage.free / 1e9
    if free_gb < 20:
        return [f'Low disk space on {check_dir} filesystem: {free_gb:.1f} GB free '
                f'-- a batch tomogram run can consume far more than this']
    return []


def _check_output_tilt_counts(expected_sections: dict, out_dir: Path) -> tuple:
    """
    Post-run sanity check: for every TS whose .aln now exists in out_dir,
    compare its total_frames (RawSize third value) against the mdoc's own
    section count recorded during preflight.

    This deliberately re-reads the structured .aln/RawSize output rather
    than scraping AreTomo3's free-text stdout log -- the log's format is
    not a stable, version-safe parsing target (ironic given this whole
    check exists because AreTomo3's *mdoc* parsing changed between
    versions), whereas .aln's RawSize field is the same well-defined
    interchange value this codebase already depends on elsewhere.

    Run this regardless of whether AreTomo3 exited cleanly or crashed --
    whichever TS did get an .aln written are worth checking either way,
    and a mismatch here is exactly the "AreTomo3 recovered the wrong
    number of tilts" signature that would flag a future parser regression
    even if it wasn't caught at the mdoc-preflight stage.

    Returns (mismatches, n_checked) where mismatches is a list of
    '{ts_name}: expected N, got M' strings.
    """
    from aretomo3_preprocess.shared.parsers import parse_aln_file

    mismatches = []
    n_checked = 0
    for ts_name, expected in expected_sections.items():
        aln_path = out_dir / f'{ts_name}.aln'
        if not aln_path.exists():
            continue
        n_checked += 1
        try:
            actual = parse_aln_file(aln_path)['total_frames']
        except Exception as e:
            mismatches.append(f'{ts_name}: expected {expected} tilts (from mdoc), '
                              f'but .aln failed to parse ({e})')
            continue
        if actual is not None and actual != expected:
            mismatches.append(f'{ts_name}: expected {expected} tilts (from mdoc), '
                              f'got {actual} (from .aln)')
    return mismatches, n_checked


def _check_completeness(expected_sections: dict, out_dir: Path) -> dict:
    """
    Post-run sanity check: cross-reference AreTomo3's own MdocDone.txt
    (the file its -Resume logic actually reads to decide what to skip --
    see the module docstring for --resume) against whether each expected
    TS actually has a non-empty primary output stack (ts-XXXX.mrc) on disk.

    Catches failures earlier than _check_output_tilt_counts can -- a TS
    that died during motion correction (before alignment ever ran) never
    gets an .aln written at all, so the tilt-count check has no way to
    see it. This only needs the main .mrc, which cmd=0 writes first, so
    it can flag a TS as incomplete even when nothing else about it exists.

    Also specifically flags the failure mode that has caused real data
    loss once already: MdocDone.txt marking a TS done while its .mrc is
    missing or 0 bytes -- AreTomo3 believing a TS finished when it didn't
    means a plain --resume would silently skip ever reprocessing it.

    Run this regardless of whether AreTomo3 exited cleanly or crashed --
    a disk-full or killed run is exactly when this inconsistency shows up.

    Returns {
        'n_checked':       int,
        'mdocdone_found':  bool,
        'not_done':        [ts_name, ...]  -- absent from MdocDone.txt; --resume will (re)do these
        'done_but_empty':  [ts_name, ...]  -- MdocDone.txt says done, but .mrc missing/0 bytes (serious)
        'empty_not_done':  [ts_name, ...]  -- .mrc missing/0 bytes but not marked done (expected, harmless)
    }
    """
    mdocdone_path = out_dir / 'MdocDone.txt'
    mdocdone_found = mdocdone_path.exists()
    done = set()
    if mdocdone_found:
        for line in mdocdone_path.read_text(errors='replace').splitlines():
            line = line.strip()
            if line.endswith('.mdoc'):
                done.add(line[:-len('.mdoc')])

    not_done, done_but_empty, empty_not_done = [], [], []
    for ts_name in expected_sections:
        mrc_path = out_dir / f'{ts_name}.mrc'
        is_empty = not mrc_path.exists() or mrc_path.stat().st_size == 0
        is_done  = ts_name in done

        if not is_done:
            not_done.append(ts_name)
        if is_empty:
            (done_but_empty if is_done else empty_not_done).append(ts_name)

    return {
        'n_checked':      len(expected_sections),
        'mdocdone_found': mdocdone_found,
        'not_done':       not_done,
        'done_but_empty': done_but_empty,
        'empty_not_done': empty_not_done,
    }


def _check_vol_completeness(ts_names: set, out_dir: Path) -> dict:
    """
    Post-run sanity check for cmd=1/2 (mrc mode): does each TS in
    ts_names have a non-empty main reconstruction (ts-XXXX_Vol.mrc)?

    Unlike _check_completeness (cmd=0/mdoc mode), there's no MdocDone.txt
    equivalent here to cross-reference against -- confirmed empirically
    against a real cmd=2 output directory that AreTomo3 writes MdocDone.txt
    unconditionally but leaves it EMPTY outside mdoc mode, so it can't be
    used as a "does AreTomo3 think this is done" signal for cmd=1/2. This
    is a direct file check only: missing/empty _Vol.mrc means that TS
    didn't finish reconstruction, full stop.

    Run this regardless of whether AreTomo3 exited cleanly or crashed.

    Returns {'n_checked': int, 'incomplete': [ts_name, ...]}.
    """
    incomplete = []
    for ts_name in ts_names:
        vol_path = out_dir / f'{ts_name}_Vol.mrc'
        if not vol_path.exists() or vol_path.stat().st_size == 0:
            incomplete.append(ts_name)
    return {'n_checked': len(ts_names), 'incomplete': sorted(incomplete)}


# ─────────────────────────────────────────────────────────────────────────────
# Live progress (polls AreTomo3's own output files -- never its stdout/stderr,
# which must stay a plain file redirect with nothing on our side reading it;
# see the note above the subprocess.Popen call in run() for why)
# ─────────────────────────────────────────────────────────────────────────────

def _read_metrics_csv(path: Path) -> list:
    """
    Parse AreTomo3's TiltSeries_Metrics.csv (written by its own
    AreTomo/CTsMetrics.cpp: one row appended per completed TS, fsync'd
    after every write -- for --cmd 0/1 only, it isn't written for cmd=2).

    AreTomo3 is actively appending to this file while we read it here;
    only keep newline-terminated lines so a read landing mid-flush can't
    hand csv.DictReader a torn last row.
    """
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return []
    complete_lines = [l for l in text.splitlines(keepends=True) if l.endswith('\n')]
    if len(complete_lines) < 2:   # header + at least one data row
        return []
    return list(csv.DictReader(complete_lines, skipinitialspace=True))


def _format_metrics_postfix(row: dict) -> dict:
    """Format the most-recently-completed TS's metrics row for a tqdm postfix."""
    def _f(key, nd=1):
        try:
            return f'{float(row[key]):.{nd}f}'
        except (KeyError, ValueError):
            return '?'
    ts_name = row.get('Tilt_Series', '?').strip()
    if ts_name.endswith('.mrc'):
        ts_name = ts_name[:-4]
    return {
        'last':    ts_name,
        'defocus': f"{_f('Defocus(A)', 0)}Å",
        'CTFres':  f"{_f('CTF_Res(A)', 1)}Å",
        'axis':    f"{_f('Tilt_Axis', 1)}°",
        'thick':   f"{_f('Thickness(Pix)', 0)}px",
    }


def _update_progress_bar(out_dir: Path, ts_names: set, cmd_num: int,
                         bar, last_n: int) -> int:
    """One progress-bar refresh pass. Returns n (for de-duping repeat calls)."""
    if cmd_num in (0, 1):
        rows = _read_metrics_csv(out_dir / 'TiltSeries_Metrics.csv')
        n = len(rows)
        if n != last_n:
            bar.n = min(n, bar.total) if bar.total is not None else n
            if rows:
                bar.set_postfix(_format_metrics_postfix(rows[-1]), refresh=False)
            bar.refresh()
    else:
        n = 0
        for ts_name in ts_names:
            vol_path = out_dir / f'{ts_name}_Vol.mrc'
            if vol_path.exists() and vol_path.stat().st_size > 0:
                n += 1
        if n != last_n:
            bar.n = min(n, bar.total) if bar.total is not None else n
            bar.refresh()
    return n


def _poll_progress_loop(stop_event: threading.Event, out_dir: Path,
                        ts_names: set, cmd_num: int, bar,
                        interval: float = 5.0) -> None:
    """
    Background-thread progress updater. Runs alongside proc.wait() in the
    main thread, polling AreTomo3's own output files on a timer -- this
    must NEVER read AreTomo3's stdout/stderr (that pipe removal is load-
    bearing, not just tidiness -- see the note above subprocess.Popen).
    A failure here must never take down the actual run, hence the blanket
    except: this is a display, not part of the pipeline.
    """
    last_n = -1
    while not stop_event.is_set():
        try:
            last_n = _update_progress_bar(out_dir, ts_names, cmd_num, bar, last_n)
        except Exception:
            pass
        stop_event.wait(interval)


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

    Returns (errors, warnings, mdoc_stats, input_ts_names) where:
      errors          — fatal, always abort
      warnings        — abort unless --force
      mdoc_stats      — dict from _check_mdocs (has 'expected_sections' for
                        the post-run tilt-count/completeness checks), or
                        None outside mdoc mode
      input_ts_names  — set of TS-name stems discovered for this run (from
                        the same file listing used for the "Input files
                        found" preflight line), for the post-run
                        completeness check outside mdoc mode (cmd=1/2)
    """
    errors     = []
    warnings   = []
    mdoc_stats = None

    print(f'  AreTomo3 binary : {_detect_aretomo3_version(args.aretomo3_bin)}')

    warnings.extend(_check_gpus(args.gpu))
    warnings.extend(_check_disk_space(Path(args.output)))

    # ── 1. Gain / fm-dose (only needed for cmd 0 = full pipeline) ─────────
    if args.gain is not None:
        if not Path(args.gain).exists():
            errors.append(f'Gain file not found: {args.gain!r}')
    elif args.cmd == 0:
        errors.append('--gain is required for --cmd 0 (motion correction mode)')

    if args.fm_dose is None and args.cmd == 0:
        errors.append('--fm-dose is required for --cmd 0 (motion correction mode)')

    # ── 2. Input files found ───────────────────────────────────────────────
    # Filter with the SAME effective skip list _build_cmd will actually pass
    # to AreTomo3 (ts-name patterns are dropped there), so preflight counts
    # and mdoc checks reflect the file set that's really about to be run.
    _effective_skips = _effective_in_skips(args.in_skips)
    _dropped_ts_skips = [s for s in (args.in_skips or [])
                         if s and s.startswith('ts-')]
    if _dropped_ts_skips:
        # True for every --cmd today: only file-type patterns (_CTF/_Vol)
        # are passed to AreTomo3's -InSkips. cmd=0/1 have no other exclusion
        # mechanism at all; cmd=2 staging accepts an in_skips parameter but
        # doesn't currently act on it either (_setup_cmd2_staging filters
        # only by --select-ts) -- so this is not cmd-specific.
        warnings.append(
            f'--in-skips {_dropped_ts_skips} — TS-name patterns have no '
            f'effect for --cmd {args.cmd} (only file-type patterns like '
            f'_CTF/_Vol are passed to -InSkips; use --select-ts to exclude '
            f'specific TS for --cmd 2). These TS will be processed.'
        )
    in_dir, pattern, mdoc_files = _find_input_files(
        args.in_prefix, args.in_suffix, _effective_skips)
    if not in_dir.is_dir() and not (args.in_suffix == 'mrc' and args.cmd == 2):
        errors.append(f'Input directory not found: {in_dir}/')
    elif not mdoc_files:
        if args.in_suffix == 'mrc' and args.cmd == 2:
            # cmd=2 dry-run: symlinks not yet created — check project.json
            proj_stacks = load_or_create().get('input_stacks', {}).get('stacks', {})
            if proj_stacks:
                print(f'  Input files     : {len(proj_stacks)} stacks in '
                      f'project.json (symlinks will be created before run)')
            else:
                errors.append(
                    f'No .mrc files found in {in_dir}/ and input_stacks not in '
                    f'project.json.\n'
                    f'       Run --cmd 0 first (registers stacks automatically), '
                    f'or register manually:\n'
                    f'         aretomo3-preprocess enrich --mrc-data <run001/>'
                )
        else:
            errors.append(
                f'No .{args.in_suffix} files found matching {in_dir}/{pattern}\n'
                f'       (run rename-ts first if using mdoc mode)'
            )
    else:
        print(f'  Input files     : {len(mdoc_files)} .{args.in_suffix} files '
              f'found ({in_dir}/{pattern})')

    # ── 2b. Mdoc parsing preflight (mdoc mode only) ───────────────────────
    if args.in_suffix == 'mdoc' and mdoc_files:
        mdoc_errors, mdoc_stats = _check_mdocs(mdoc_files)
        if mdoc_errors:
            errors.extend(mdoc_errors)

        n = mdoc_stats['n_mdocs']
        v222, v230 = mdoc_stats['n_v222'], mdoc_stats['n_v230']
        print(f'  Mdoc parsing    : {v230}/{n} OK for AreTomo3 >=2.3.0'
              f'  |  {v222}/{n} OK for AreTomo3 <2.3.0')
        if v222 != v230:
            print(f'                    ** {abs(v222 - v230)} mdoc(s) parse differently '
                  f'depending on AreTomo3 version -- '
                  f'{"would break on upgrade to" if v222 > v230 else "require"} 2.3.0 **')

        if mdoc_stats['median_tilts'] is not None:
            tilt_range = (f'  (range {mdoc_stats["min_tilts"]}-{mdoc_stats["max_tilts"]})'
                          if mdoc_stats['min_tilts'] != mdoc_stats['max_tilts'] else '')
            fpt = (f'  |  {mdoc_stats["median_frames_per_tilt"]} frames/tilt (median)'
                  if mdoc_stats['median_frames_per_tilt'] is not None else '')
            print(f'                    {mdoc_stats["n_valid"]}/{n} tilts recovered OK'
                  f'  |  {mdoc_stats["median_tilts"]} tilts/TS (median){tilt_range}{fpt}')

        # All-mdocs internal consistency (not just the first file against
        # --apix/--kv below): the mdoc's own PixelSpacing is uncalibrated and
        # routinely overridden by --apix on purpose, so disagreement with
        # --apix is expected and not itself a red flag. But every mdoc in
        # one batch disagreeing with EACH OTHER means a mixed-session
        # dataset (different mag/settings), which is worth flagging.
        ps_values, kv_values = set(), set()
        for path in mdoc_files:
            meta = _read_mdoc_metadata(path)
            if meta.get('PixelSpacing') is not None:
                ps_values.add(round(meta['PixelSpacing'], 4))
            if meta.get('Voltage') is not None:
                kv_values.add(round(meta['Voltage'], 1))
        if len(ps_values) > 1:
            warnings.append(
                f'Mdocs disagree on PixelSpacing (uncalibrated): {sorted(ps_values)} '
                f'-- mixed-session dataset?'
            )
        if len(kv_values) > 1:
            warnings.append(
                f'Mdocs disagree on Voltage: {sorted(kv_values)} kV -- mixed-session dataset?'
            )

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
    # load_or_create() only raises SystemExit for a genuinely serious problem
    # (working_dir mismatch or corrupt JSON, with its own clear error message
    # already printed) -- let it propagate and abort here, at the cheap
    # preflight stage, rather than silently continuing on a project.json we
    # can't trust and only discovering the same failure after AreTomo3 has
    # already run for hours (see the final save below).
    project = load_or_create()
    gc = project.get('gain_check', {})

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

    input_ts_names = {f.stem for f in mdoc_files} if mdoc_files else set()

    return errors, warnings, mdoc_stats, input_ts_names


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
    ali.add_argument('--sart', type=int, nargs=2, default=None,
                     metavar=('ITERS', 'PROJS_PER_SUBSET'),
                     help='SART reconstruction parameters: number of iterations '
                          'and projections per subset (AreTomo3 defaults: 20 5). '
                          'Only used when --wbp 0. Omit to use AreTomo3 defaults. (-Sart)')
    ali.add_argument('--flip-vol', type=int, default=1,
                     help='Flip reconstructed volume (-FlipVol)')
    ali.add_argument('--tilt-cor', nargs='+', default=['1'],
                     metavar=('ENABLE', 'ANGLE'),
                     help='Tilt correction: 0=off, 1=auto-correct. Optionally '
                          'append a fixed offset in degrees, e.g. --tilt-cor 1 2.5 '
                          '(-TiltCor)')
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
    # Force line-buffered stdout so output appears immediately when piped (e.g. | tee)
    sys.stdout.reconfigure(line_buffering=True)

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
    #        (from project.json input_stacks) are created directly in --output/.
    #        AreTomo3 is pointed at --output/.  src_in_dir (e.g. run003) is
    #        NEVER modified.
    selected_ts = None
    if args.in_suffix == 'mrc' and args.cmd == 2:
        selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
        staging_dir = _setup_cmd2_staging(
            out_dir     = out_dir,
            src_in_dir  = src_in_dir,
            in_skips    = args.in_skips,
            dry_run     = args.dry_run,
            selected_ts = selected_ts,
            split_sum   = args.split_sum,
        )
        stem = Path(args.in_prefix).name      # e.g. 'ts-'
        args.in_prefix = str(staging_dir / stem)

    in_dir = Path(args.in_prefix).parent  # staging_dir for cmd=2, src_in_dir otherwise

    # ── Pre-flight validation ──────────────────────────────────────────────
    print('Pre-flight checks:')
    errors, warnings, mdoc_stats, input_ts_names = _validate(args)

    # _setup_cmd2_staging only ever ADDS symlinks, never removes ones left
    # over from a prior run into the same --output with a broader (or no)
    # --select-ts -- so input_ts_names (globbed from the staging dir) can
    # include stale, no-longer-selected TS. Filter against the CURRENT
    # selection so the progress bar / completeness check "expected" count
    # reflects what's actually being run, not stale staging leftovers.
    if selected_ts is not None:
        input_ts_names = input_ts_names & selected_ts

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

    _print_grouped_params(cmd)

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
    print('Running AreTomo3...\n')

    # Redirect AreTomo3's stdout/stderr straight to the log file's underlying
    # fd — the same mechanism as `... > log_path 2>&1` from a shell — instead
    # of going through any pipe (Python-side draining, or a `tee` subprocess).
    #
    # Both of those earlier approaches still put a *pipe* between AreTomo3
    # and the log file, and a pipe has a small, finite kernel buffer (64KB on
    # Linux) that needs an active reader to keep draining it; if the reader
    # doesn't get scheduled promptly, AreTomo3's own write() calls can block.
    # A plain file redirect has no such reader dependency — writes land in
    # the page cache and effectively never block. That's a structural
    # difference from every "direct" AreTomo3 invocation we've ever run
    # (always `> log 2>&1`, never piped), and per the 2026-07 investigation
    # it's suspected to be *why* wrapper-launched runs crash far more often
    # than direct ones on marginal/low-signal data: AreTomo3 reseeds an
    # internal RNG from the wall clock on every refinement iteration
    # (MaUtil/GGenRandoms.cu), and pipe-induced stalls are a plausible way to
    # perturb exactly when those reseeds land relative to real time.
    #
    # Live progress below is a SEPARATE mechanism from that log redirect —
    # a background thread polling AreTomo3's own output files (never its
    # stdout/stderr) on a timer, so it carries none of the pipe-stall risk
    # above. See _poll_progress_loop.
    total_ts = len(input_ts_names) if input_ts_names else None
    progress_bar = tqdm(total=total_ts, desc='AreTomo3', unit='TS')
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_progress_loop,
        args=(stop_event, out_dir, input_ts_names, args.cmd, progress_bar),
        daemon=True,
    )
    poll_thread.start()
    try:
        with open(log_path, 'wb') as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            returncode = proc.wait()
    finally:
        stop_event.set()
        poll_thread.join(timeout=2)
        try:
            _update_progress_bar(out_dir, input_ts_names, args.cmd, progress_bar, -1)
        except Exception:
            pass
        progress_bar.close()
    log_lines = log_path.read_text(errors='replace').splitlines(keepends=True)

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

    # ── Post-run tilt-count check (mdoc mode only; runs whether AreTomo3
    # succeeded or crashed -- whatever .aln files exist are worth checking) ──
    if mdoc_stats is not None:
        mismatches, n_checked = _check_output_tilt_counts(
            mdoc_stats['expected_sections'], out_dir)
        if n_checked:
            print(f'\nTilt-count check: {n_checked - len(mismatches)}/{n_checked} '
                  f'TS match mdoc-expected tilt count')
            if mismatches:
                print(f'  ** {len(mismatches)} TS recovered a different tilt count '
                      f'than the mdoc has -- possible parser regression: **')
                for line in mismatches[:20]:
                    print(f'    {line}')
                if len(mismatches) > 20:
                    print(f'    ... ({len(mismatches) - 20} more)')

    # ── Post-run completeness check (mdoc mode only; runs whether AreTomo3
    # succeeded or crashed) -- the tilt-count check above only sees TS that
    # got far enough to have an .aln; a TS that died during motion
    # correction (before alignment ever ran) never gets one, so it's
    # otherwise invisible. This also catches AreTomo3's own MdocDone.txt
    # claiming a TS is done while its output is empty -- one exit code
    # covers the whole batch, so it can't tell you which of potentially
    # hundreds of TS actually finished; this can. ──
    if mdoc_stats is not None:
        completeness = _check_completeness(mdoc_stats['expected_sections'], out_dir)
        if completeness['mdocdone_found']:
            n_total, n_not_done = completeness['n_checked'], len(completeness['not_done'])
            print(f'\nCompleteness check: {n_total - n_not_done}/{n_total} TS marked '
                  f'done in MdocDone.txt'
                  + (f'  ({n_not_done} not done -- --resume will (re)process these)'
                     if n_not_done else ''))
            if completeness['done_but_empty']:
                print(f'  ** {len(completeness["done_but_empty"])} TS marked done in '
                      f'MdocDone.txt but have a missing/empty .mrc -- AreTomo3 believes '
                      f'these finished when they did not; a plain --resume will WRONGLY '
                      f'skip reprocessing them: **')
                for ts in completeness['done_but_empty'][:20]:
                    print(f'    {ts}')
                if len(completeness['done_but_empty']) > 20:
                    print(f'    ... ({len(completeness["done_but_empty"]) - 20} more)')
            if completeness['empty_not_done']:
                print(f'  {len(completeness["empty_not_done"])} TS have a missing/empty '
                      f'.mrc and are not marked done -- --resume will reprocess these '
                      f'normally, no action needed')

    # ── Post-run completeness check (cmd=1/2 only; runs whether AreTomo3
    # succeeded or crashed) -- no MdocDone.txt-equivalent ledger exists
    # outside mdoc mode (confirmed empirically: AreTomo3 writes MdocDone.txt
    # unconditionally but leaves it empty for cmd=1/2), so this is a direct
    # "does the reconstructed volume exist and have data" check instead. ──
    elif args.cmd in (1, 2) and input_ts_names:
        vol_check = _check_vol_completeness(input_ts_names, out_dir)
        n_total = vol_check['n_checked']
        n_incomplete = len(vol_check['incomplete'])
        print(f'\nCompleteness check: {n_total - n_incomplete}/{n_total} TS have a '
              f'non-empty _Vol.mrc')
        if vol_check['incomplete']:
            print(f'  {n_incomplete} TS have a missing/empty _Vol.mrc:')
            for ts in vol_check['incomplete'][:20]:
                print(f'    {ts}')
            if n_incomplete > 20:
                print(f'    ... ({n_incomplete - 20} more)')

    print()
    if returncode != 0:
        print(f'ERROR: AreTomo3 exited with code {returncode}')
        print('Project JSON not updated (recording only successful runs).')
        sys.exit(returncode)

    print(f'AreTomo3 finished successfully.')

    # AreTomo3 has already finished successfully at this point -- if
    # project.json bookkeeping fails (e.g. corrupted or working-dir-mismatched
    # mid-run), don't let that surface as a bare, unexplained traceback that
    # makes it look like the whole run failed. Print what would have been
    # saved so nothing is lost, then exit non-zero so the failure still
    # propagates to any calling script.
    try:
        # ── Register cmd=0 output stacks and TLT dir for later runs ──────
        if args.cmd == 0:
            register_input_stacks(out_dir, in_skips=args.in_skips, tlt_dir=out_dir)

        # ── Save to project JSON ──────────────────────────────────────────
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
    except SystemExit:
        print()
        print('ERROR: AreTomo3 finished successfully, but the run could not be '
              'recorded in aretomo3_project.json (see error above).')
        print(f'       output_dir: {out_dir.resolve()}')
        print(f'       log:        {log_path.resolve()}')
        print(f'       returncode: {returncode}')
        print('       Fix project.json (see recovery steps above), then re-run '
              'this exact command -- it will pick up the existing AreTomo3 output.')
        raise
