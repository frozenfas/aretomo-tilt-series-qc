"""
run-aretomo3 — wrapper around the AreTomo3 binary for batch tilt-series processing.

Builds the AreTomo3 command from Python CLI arguments, prints it as a formatted
multi-line shell snippet with inline annotations, then runs it with live
stdout/stderr streaming captured to a log file.

Before running, validates that:
  - the gain reference file exists
  - input mdoc/mrc files are found matching the prefix/suffix pattern
  - gain orientation (--flip-gain / --rot-gain) matches the recommendation in
    aretomo3_project.json from a prior check-gain-transform run
  - pixel spacing (--apix) matches PixelSpacing in the mdoc files
  - voltage (--kv) matches Voltage in the mdoc files (if present)

Parameter mismatches are reported as warnings that block execution unless
--force is given.  Missing files are always fatal.

On success, saves the invocation to aretomo3_project.json under the
'run_aretomo3' key with a backup in the output directory.

Typical usage
-------------
    aretomo3-preprocess run-aretomo3 \\
        --output run001 \\
        --gain gain_20260213T101027.mrc \\
        --apix 1.63 \\
        --fm-dose 0.52 \\
        --gpu 2 3 \\
        --flip-gain 1 \\
        --dry-run
"""

import re
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import (
    load_or_create, update_section, args_to_dict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Brief annotations for each AreTomo3 flag (shown in dry-run output)
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_COMMENTS = {
    '-InPrefix':     'input file prefix for batch discovery',
    '-InSuffix':     'input file suffix (mdoc=live pipeline, mrc=offline)',
    '-InSkips':      'filename patterns to skip',
    '-OutDir':       'output directory',
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
    if args.in_skips is not None:
        cmd += ['-InSkips'] + list(args.in_skips)

    cmd += ['-OutDir', args.output]
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


def _find_input_files(in_prefix: str, in_suffix: str) -> tuple:
    """Return (in_dir, pattern, sorted file list).  in_dir is a Path."""
    p = Path(in_prefix)
    in_dir = p.parent
    stem   = p.name
    pattern = f'{stem}*.{in_suffix}'
    files = sorted(in_dir.glob(pattern)) if in_dir.is_dir() else []
    return in_dir, pattern, files


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
    in_dir, pattern, mdoc_files = _find_input_files(args.in_prefix, args.in_suffix)
    if not in_dir.is_dir():
        errors.append(f'Input directory not found: {in_dir}/')
    elif not mdoc_files:
        errors.append(
            f'No .{args.in_suffix} files found matching {in_dir}/{pattern}\n'
            f'       (run rename-ts first if using mdoc mode)'
        )
    else:
        print(f'  Input files     : {len(mdoc_files)} .{args.in_suffix} files '
              f'found ({in_dir}/{pattern})')

    # ── 3. Gain transform vs gain_check JSON (only if gain provided) ──────
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
    req.add_argument('--apix', '-a', type=float, required=True,
                     help='Pixel size of input movies in Å/px (-PixSize)')
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
    inp.add_argument('--in-skips', nargs='*', default=None, metavar='PATTERN',
                     help='Filename patterns to exclude (-InSkips); '
                          'flag omitted if not given')
    inp.add_argument('--serial', type=int, default=1,
                     help='Seconds to wait for next series; 1=offline batch (-Serial)')

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
    ali.add_argument('--tilt-axis', type=float, nargs='+', default=None,
                     metavar='ANGLE',
                     help='Tilt axis angle [REFINE_FLAG]; auto-detected if omitted (-TiltAxis)')
    ali.add_argument('--align-z', type=int, default=None,
                     help='Sample thickness for alignment in pixels; '
                          'auto-estimated if omitted (-AlignZ)')
    ali.add_argument('--group', type=int, nargs=2, default=None,
                     metavar=('GLOBAL', 'LOCAL'),
                     help='Frame grouping GLOBAL LOCAL; AreTomo3 default if omitted (-Group)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--aretomo3', dest='aretomo3_bin', default='AreTomo3',
                     help='Path to or name of the AreTomo3 executable')
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
    out_dir = Path(args.output)

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
