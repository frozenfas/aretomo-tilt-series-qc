"""
run-aretomo3 — wrapper around the AreTomo3 binary for batch tilt-series processing.

Builds the AreTomo3 command from Python CLI arguments, prints it as a formatted
multi-line shell snippet, then runs it with live stdout/stderr streaming.

On success, saves the invocation and key arguments to aretomo3_project.json
under the 'run_aretomo3' key (backup written to the output directory).

Defaults are tuned for offline batch processing of K3 TIFF movies (-Cmd 0,
-Serial 1). For EER movies: set --fm-int to the number of raw frames per
rendered frame (typically 15) and consider --eer-sampling 2 --mc-bin 2.

Typical usage
-------------
    aretomo3-preprocess run-aretomo3 \\
        --output run001 \\
        --gain gain_20260213T101027.mrc \\
        --apix 1.63 \\
        --fm-dose 0.52 \\
        --gpu 2 3 \\
        --flip-gain 1

Use --dry-run to preview the command without executing.
"""

import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num(x) -> str:
    """Format a number, stripping trailing .0 from whole-number floats."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def _is_flag(token: str) -> bool:
    """True if token is an AreTomo3 flag (e.g. -InPrefix), not a negative number."""
    return len(token) >= 2 and token[0] == '-' and token[1].isalpha()


def _fmt_command(cmd: list) -> str:
    """Format a flat command list as a multi-line shell snippet.

    Each flag and its associated value(s) appear on one indented line,
    joined with backslash continuation.  Negative numbers (e.g. -28.0 for
    tilt angles) are correctly grouped with their preceding flag rather than
    being treated as new flags.
    """
    groups = [[cmd[0]]]
    for token in cmd[1:]:
        if _is_flag(token):
            groups.append([token])
        else:
            groups[-1].append(token)
    return ' \\\n    '.join(' '.join(g) for g in groups)


# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_cmd(args) -> list:
    """Assemble the AreTomo3 command as a flat list of string tokens."""
    cmd = [args.aretomo3_bin]

    # Input / discovery
    cmd += ['-InPrefix', args.in_prefix]
    cmd += ['-InSuffix', args.in_suffix]
    if args.in_skips is not None:
        cmd += ['-InSkips'] + list(args.in_skips)

    # Output
    cmd += ['-OutDir', args.output]

    # Gain
    cmd += ['-Gain',     args.gain]
    cmd += ['-FlipGain', _num(args.flip_gain)]
    cmd += ['-RotGain',  _num(args.rot_gain)]

    # GPU and pipeline mode
    cmd += ['-Gpu'] + [_num(g) for g in args.gpu]
    cmd += ['-Cmd',    _num(args.cmd)]
    cmd += ['-Serial', _num(args.serial)]

    # Microscope / CTF
    cmd += ['-PixSize',     _num(args.apix)]
    cmd += ['-Kv',          _num(args.kv)]
    cmd += ['-Cs',          _num(args.cs)]
    cmd += ['-AmpContrast', _num(args.amp_contrast)]

    # Acquisition / motion correction
    cmd += ['-FmDose',  _num(args.fm_dose)]
    cmd += ['-McBin',   _num(args.mc_bin)]
    cmd += ['-McPatch'] + [_num(v) for v in args.mc_patch]
    cmd += ['-FmInt',   _num(args.fm_int)]
    if args.eer_sampling is not None:
        cmd += ['-EerSampling', _num(args.eer_sampling)]
    cmd += ['-SplitSum', _num(args.split_sum)]

    # Reconstruction
    cmd += ['-VolZ',     _num(args.vol_z)]
    cmd += ['-AtBin']  + [_num(v) for v in args.at_bin]
    cmd += ['-AtPatch'] + [_num(v) for v in args.at_patch]
    cmd += ['-Wbp',      _num(args.wbp)]
    cmd += ['-FlipVol',  _num(args.flip_vol)]

    # Alignment
    cmd += ['-TiltCor', _num(args.tilt_cor)]
    cmd += ['-DarkTol', _num(args.dark_tol)]
    if args.tilt_axis is not None:
        cmd += ['-TiltAxis'] + [_num(v) for v in args.tilt_axis]
    if args.align_z is not None:
        cmd += ['-AlignZ', _num(args.align_z)]
    if args.group is not None:
        cmd += ['-Group'] + [_num(v) for v in args.group]

    # CTF correction and output files
    cmd += ['-CorrCTF', _num(args.corr_ctf)]
    cmd += ['-OutXF',   _num(args.out_xf)]
    cmd += ['-OutImod', _num(args.out_imod)]

    return cmd


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
    req.add_argument('--gain', '-g', required=True,
                     help='Gain reference file (.mrc or .gain) (-Gain)')
    req.add_argument('--apix', '-a', type=float, required=True,
                     help='Pixel size of input movies in Å/px (-PixSize)')
    req.add_argument('--fm-dose', '-d', type=float, required=True,
                     help='Electron dose per raw frame in e⁻/Å² (-FmDose)')
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
                          'flag omitted from command if not given')
    inp.add_argument('--serial', type=int, default=1,
                     help='Seconds to wait for next series in live mode; '
                          '1 = offline batch (-Serial)')

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
                    help='Raw frames per rendered frame: 1 for TIFF, ~15 for EER (-FmInt)')
    mc.add_argument('--eer-sampling', type=int, default=None,
                    help='EER sampling rate; omitted from command if not given (-EerSampling)')
    mc.add_argument('--split-sum', type=int, default=1,
                    help='Write half-sums for dose weighting: 1=yes 0=no (-SplitSum)')

    ali = p.add_argument_group('alignment and reconstruction')
    ali.add_argument('--vol-z', type=int, default=2046,
                     help='Reconstruction Z thickness in pixels; 0=auto-estimate (-VolZ)')
    ali.add_argument('--at-bin', type=float, nargs='+', default=[4.0],
                     metavar='BIN',
                     help='AreTomo binning factor(s); up to 3 values for '
                          'multi-resolution output (-AtBin)')
    ali.add_argument('--at-patch', type=int, nargs=2, default=[0, 0],
                     metavar=('X', 'Y'),
                     help='Local alignment patch grid; 0 0=global only (-AtPatch X Y)')
    ali.add_argument('--wbp', type=int, default=1,
                     help='Reconstruction method: 1=WBP 0=SART (-Wbp)')
    ali.add_argument('--flip-vol', type=int, default=1,
                     help='Flip reconstructed volume for conventional orientation (-FlipVol)')
    ali.add_argument('--tilt-cor', type=int, default=1,
                     help='Apply tilt angle offset correction (-TiltCor)')
    ali.add_argument('--dark-tol', type=float, default=0.7,
                     help='Dark frame rejection tolerance (-DarkTol)')
    ali.add_argument('--corr-ctf', type=int, default=1,
                     help='Local CTF correction before reconstruction: 1=yes 0=no (-CorrCTF)')
    ali.add_argument('--out-xf', type=int, default=1,
                     help='Write IMOD-compatible XF alignment files (-OutXF)')
    ali.add_argument('--out-imod', type=int, default=1,
                     help='Write IMOD support files for RELION (-OutImod)')
    # Optional with no default — omitted from AreTomo3 command if not supplied
    ali.add_argument('--tilt-axis', type=float, nargs='+', default=None,
                     metavar='ANGLE',
                     help='Tilt axis angle [REFINE_FLAG]; auto-detected if omitted (-TiltAxis)')
    ali.add_argument('--align-z', type=int, default=None,
                     help='Sample thickness for alignment in pixels; '
                          'auto-estimated if omitted (-AlignZ)')
    ali.add_argument('--group', type=int, nargs=2, default=None,
                     metavar=('GLOBAL', 'LOCAL'),
                     help='Frame grouping for global and local motion measurement; '
                          'AreTomo3 default if omitted (-Group)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--aretomo3', dest='aretomo3_bin', default='AreTomo3',
                     help='Path to or name of the AreTomo3 executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print the formatted command without executing')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output)

    # Check binary exists (skip in dry-run so previewing on non-GPU machines works)
    if not args.dry_run:
        if shutil.which(args.aretomo3_bin) is None and not Path(args.aretomo3_bin).is_file():
            print(f'ERROR: AreTomo3 binary not found: {args.aretomo3_bin!r}')
            print('       Ensure it is on PATH or use --aretomo3 /path/to/AreTomo3')
            sys.exit(1)

    cmd = _build_cmd(args)

    prefix = '[DRY RUN] ' if args.dry_run else ''
    print(f'{prefix}AreTomo3 command:\n')
    print(_fmt_command(cmd))
    print()

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory : {out_dir}')
    print('Running AreTomo3 (output streaming live)...\n')

    result = subprocess.run(cmd)

    print()
    if result.returncode != 0:
        print(f'ERROR: AreTomo3 exited with code {result.returncode}')
        print('Project JSON not updated (recording only successful runs).')
        sys.exit(result.returncode)

    print('AreTomo3 finished successfully.')

    update_section(
        section='run_aretomo3',
        values={
            'command':    ' '.join(cmd),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'output_dir': str(out_dir.resolve()),
            'returncode': result.returncode,
        },
        backup_dir=out_dir,
    )
