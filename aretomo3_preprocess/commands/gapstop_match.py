"""
gapstop-match — batch template matching with gapstop on AreTomo3 tomograms.

gapstop is a GPU-accelerated template matching tool using the STOPGAP per-tilt
wedge model implemented in JAX.

- Paper: https://www.nature.com/articles/s41467-024-47839-8
- GitLab: https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm

gapstop must be installed in the gapstop conda environment at
/opt/miniconda3/envs/gapstop/ (default), or be available on PATH.
starfile must also be installed in that environment.

Typical usage
-------------
  # Dry run to check generated STAR files and command
  aretomo3-preprocess gapstop-match \\
      --input run002-cmd2-sart-thr80 \\
      --template ribosome_14A.mrc \\
      --mask ribosome_mask.mrc \\
      --angincr 3.5 --angiter 52 --phi-angincr 3.5 --phi-angiter 52 \\
      --lp-rad 0.35 --hp-rad 0.02 \\
      --voltage 300 --amplitude-contrast 0.07 --spherical-aberration 2.7 \\
      --output gapstop_match \\
      --dry-run

  # Run on all tomograms (uses all 4 GPUs via -n 4 tiling)
  aretomo3-preprocess gapstop-match \\
      --input run002-cmd2-sart-thr80 \\
      --template ribosome_14A.mrc \\
      --mask ribosome_mask.mrc \\
      --angincr 3.5 --angiter 52 --phi-angincr 3.5 --phi-angiter 52 \\
      --lp-rad 0.35 --hp-rad 0.02 \\
      --n-tiles 4 \\
      --output gapstop_match

  # Selected TS only, with binary mask from slabify
  aretomo3-preprocess gapstop-match \\
      --input run002-cmd2-sart-thr80 \\
      --template ribosome_14A.mrc --mask ribosome_mask.mrc \\
      --angincr 3.5 --angiter 52 --phi-angincr 3.5 --phi-angiter 52 \\
      --lp-rad 0.35 --hp-rad 0.02 \\
      --select-ts run002_analysis/ts-select.csv \\
      --bmask-dir run002-cmd2-sart-thr80/slabify \\
      --bmask-suffix _mask \\
      --output gapstop_match
"""

import re
import sys
import struct
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

import numpy as np

from aretomo3_preprocess.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file,
)
from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir
from aretomo3_preprocess.shared.volume_qc import (
    central_slab_projection, projection_to_b64png,
    make_comparison_html,
)

_GAPSTOP_BIN = '/opt/miniconda3/envs/gapstop/bin/gapstop'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_gapstop(gapstop_dir=None):
    """Return path to gapstop binary, or None if not found."""
    candidates = []
    if gapstop_dir:
        candidates.append(str(Path(gapstop_dir) / 'gapstop'))
    candidates.append(_GAPSTOP_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('gapstop')


def _mrc_dims(mrc_path):
    """Read (nx, ny, nz) from MRC header without mrcfile."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)


def _mrc_angpix(mrc_path):
    """Read pixel size (Å/px) from MRC header cell_a.x / nx."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(1024)
    nx = struct.unpack_from('<i', hdr, 0)[0]
    cell_x = struct.unpack_from('<f', hdr, 40)[0]  # bytes 40-43 = xlen
    if nx > 0 and cell_x > 0:
        return cell_x / nx
    return None


def _find_volumes(in_dir, vol_suffix=None):
    """Return sorted list of (prefix, vol_path) tuples."""
    if vol_suffix:
        vol_glob = f'ts-*{vol_suffix}_Vol.mrc'
    else:
        vol_glob = 'ts-*_Vol.mrc'

    vols = [v for v in sorted(in_dir.glob(vol_glob))
            if '_EVN' not in v.name and '_ODD' not in v.name]

    if not vols and not vol_suffix:
        vols = [v for v in sorted(in_dir.glob('ts-*.mrc'))
                if not any(t in v.name for t in ('_EVN', '_ODD', '_CTF'))]

    def _prefix(v):
        name = v.stem
        for tag in ('_Vol', vol_suffix or ''):
            if tag and name.endswith(tag):
                name = name[:-len(tag)]
        return name

    return [(_prefix(v), v) for v in vols]


def _read_ts_metadata(aretomo_dir, prefix, dose_override=None):
    """
    Read tilt angles, defocus, and cumulative prior dose for one TS prefix.

    Returns
    -------
    tilt_angles  : np.ndarray  tilt-sorted corrected tilts (degrees)
    defocus_df   : dict or None  {sec: entry} from parse_ctf_file, or None
    exposure_arr : np.ndarray  cumulative prior dose (e⁻/Å²), RELION convention
    frames       : list  aligned frames from .aln (tilt order, dark excluded)
    """
    aretomo_dir = Path(aretomo_dir)
    aln_path = aretomo_dir / f'{prefix}.aln'
    ctf_path = aretomo_dir / f'{prefix}_CTF.txt'
    tlt_path = aretomo_dir / f'{prefix}_TLT.txt'

    if not aln_path.exists():
        raise FileNotFoundError(f'{prefix}: .aln not found at {aln_path}')
    if not tlt_path.exists():
        raise FileNotFoundError(f'{prefix}: _TLT.txt not found at {tlt_path}')

    aln = parse_aln_file(aln_path)
    tlt = parse_tlt_file(tlt_path)

    frames = aln['frames']
    if not frames:
        raise ValueError(f'{prefix}: no aligned frames in .aln')

    tilt_angles = np.array([f['tilt'] for f in frames])

    # CTF (optional)
    defocus_df = None
    if ctf_path.exists():
        defocus_df = parse_ctf_file(ctf_path)

    # Cumulative prior dose (RELION convention: first frame = prior dose 0)
    sorted_rows = sorted(tlt.values(), key=lambda r: r['acq_order'])
    cum = 0.0
    acq_to_prior = {}
    for row in sorted_rows:
        acq_to_prior[row['acq_order']] = round(cum, 2)
        per_frame = dose_override if dose_override is not None else row['dose_e_per_A2']
        cum += per_frame

    exposure_arr = np.array([
        acq_to_prior[tlt[f['sec']]['acq_order']] for f in frames
    ])

    return tilt_angles, defocus_df, exposure_arr, frames


def _write_wedge_list(out_path, tomo_num, angpix, nx, ny, nz,
                      tilt_angles, defocus_df, frames, exposure_arr, args):
    """
    Write a STOPGAP-format wedge list STAR file.

    Defocus values from AreTomo3 _CTF.txt are in Å; converted to µm here.
    Defocus1/defocus2 are used if available; phase shift included if non-zero.
    """
    try:
        import pandas as pd
        import starfile
    except ImportError as e:
        raise ImportError(
            'starfile and pandas are required for wedge list generation.\n'
            f'  /opt/miniconda3/envs/gapstop/bin/pip install starfile'
        ) from e

    n = len(tilt_angles)

    data = {
        'tomo_num':     [tomo_num] * n,
        'pixelsize':    [angpix] * n,
        'tomo_x':       [nx] * n,
        'tomo_y':       [ny] * n,
        'tomo_z':       [nz] * n,
        'z_shift':      [0.0] * n,
        'tilt_angle':   list(tilt_angles),
        'voltage':      [args.voltage] * n,
        'amp_contrast': [args.amplitude_contrast] * n,
        'cs':           [args.spherical_aberration] * n,
    }

    if defocus_df is not None:
        d1_um, d2_um, pshift = [], [], []
        for f in frames:
            entry = defocus_df.get(f['sec'])
            if entry:
                d1_um.append(entry['defocus1_A'] / 1e4)
                d2_um.append(entry['defocus2_A'] / 1e4)
                pshift.append(entry['phase_shift_rad'])
            else:
                d1_um.append(float('nan'))
                d2_um.append(float('nan'))
                pshift.append(0.0)
        data['defocus1'] = d1_um
        data['defocus2'] = d2_um
        data['pshift']   = pshift

    data['exposure'] = list(exposure_arr)

    df = pd.DataFrame(data)
    starfile.write({'stopgap_wedgelist': df}, str(out_path))
    return out_path


def _write_gapstop_params(param_path, tomo_path, tomo_num, wedge_path,
                           out_subdir, args, bmask=None):
    """Write a gapstop parameter STAR file for one tomogram."""
    try:
        import pandas as pd
        import starfile
    except ImportError as e:
        raise ImportError(
            'starfile and pandas are required.\n'
            f'  /opt/miniconda3/envs/gapstop/bin/pip install starfile'
        ) from e

    params = {
        'rootdir':        [str(Path('.').resolve())],
        'outputdir':      [str(out_subdir)],
        'tomo_name':      [str(tomo_path)],
        'tomo_num':       [tomo_num],
        'vol_ext':        ['.mrc'],
        'tomo_mask_name': [str(bmask) if bmask else ''],
        'wedgelist_name': [str(wedge_path)],
        'tmpl_name':      [str(Path(args.template).resolve())],
        'mask_name':      [str(Path(args.mask).resolve())],
        'symmetry':       [args.symmetry],
        'anglist_name':   [''],
        'anglist_order':  [args.anglist_order],
        'angincr':        [args.angincr],
        'angiter':        [args.angiter],
        'phi_angincr':    [args.phi_angincr],
        'phi_angiter':    [args.phi_angiter],
        'tilelist_name':  [''],
        'smap_name':      ['scores'],
        'omap_name':      ['angles'],
        'tmap_name':      ['noise'],
        'lp_rad':         [args.lp_rad],
        'lp_sigma':       [args.lp_sigma],
        'hp_rad':         [args.hp_rad],
        'hp_sigma':       [args.hp_sigma],
        'binning':        [args.binning],
        'calc_exp':       [args.calc_exp],
        'calc_ctf':       [args.calc_ctf],
        'apply_laplacian':[args.apply_laplacian],
        'noise_corr':     [args.noise_corr],
        'fourier_crop':   [args.fourier_crop],
        'scoring_fcn':    [args.scoring_fcn],
        'write_raw':      [args.write_raw],
        'tiling':         [args.tiling],
    }

    import pandas as pd
    df = pd.DataFrame(params)
    starfile.write(df, str(param_path))
    return param_path


def _print_cmd(cmd):
    """Print command multi-line, one flag+value per line."""
    it = iter(cmd)
    lines = ['  $ ' + next(it)]
    for tok in it:
        if tok.startswith('-'):
            lines.append('      ' + tok)
        else:
            lines[-1] += '  ' + tok
    print(' \\\n'.join(lines))


def _find_score_map(ts_out):
    """Find the score MRC map in the gapstop output directory."""
    candidates = sorted(ts_out.glob('scores_*.mrc'))
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'gapstop-match',
        help='Batch template matching with gapstop on AreTomo3 tomograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='AreTomo3 output directory containing ts-*_Vol.mrc files')
    inp.add_argument('--vol-suffix', default=None,
                     help='Extra suffix before _Vol.mrc (e.g. "_SART" for ts-001_SART_Vol.mrc)')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only process selected TS')
    inp.add_argument('--include', nargs='+',
                     help='Process only these TS prefixes (wildcards supported)')
    inp.add_argument('--exclude', nargs='+',
                     help='Exclude these TS prefixes (wildcards supported)')
    inp.add_argument('--bmask-dir', default=None,
                     help='Directory of per-TS binary mask MRCs (e.g. slabify output)')
    inp.add_argument('--bmask-suffix', default='',
                     help='Filename suffix for mask MRCs; e.g. "_mask" → ts-001_mask.mrc '
                          '(default: "" → ts-001.mrc)')
    inp.add_argument('--dose', type=float, default=None,
                     help='Per-tilt dose (e⁻/Å²); if omitted, reads from _TLT.txt')

    tmpl = p.add_argument_group('template matching (required)')
    tmpl.add_argument('--template', '-t', required=True,
                      help='Template MRC file')
    tmpl.add_argument('--mask', '-m', required=True,
                      help='Mask MRC file')

    ang = p.add_argument_group('angular search')
    ang.add_argument('--angincr', type=float, required=True,
                     help='Angular increment (degrees) for the primary search axis')
    ang.add_argument('--angiter', type=float, required=True,
                     help='Angular iterations for the primary axis '
                          '(search range ≈ angincr × angiter)')
    ang.add_argument('--phi-angincr', type=float, required=True,
                     help='Angular increment (degrees) for the phi (azimuth) axis')
    ang.add_argument('--phi-angiter', type=float, required=True,
                     help='Angular iterations for the phi axis')
    ang.add_argument('--anglist-order', default='zxz',
                     help='Euler angle convention for angular search (default: zxz)')
    ang.add_argument('--symmetry', default='C1',
                     help='Particle symmetry (e.g. C1, C2, C6)')

    flt = p.add_argument_group('filter parameters')
    flt.add_argument('--lp-rad', type=float, required=True,
                     help='Low-pass filter radius (Nyquist fraction, e.g. 0.35)')
    flt.add_argument('--lp-sigma', type=float, default=3.0,
                     help='Low-pass filter sigma (Nyquist fraction)')
    flt.add_argument('--hp-rad', type=float, required=True,
                     help='High-pass filter radius (Nyquist fraction, e.g. 0.02)')
    flt.add_argument('--hp-sigma', type=float, default=2.0,
                     help='High-pass filter sigma (Nyquist fraction)')

    ctf = p.add_argument_group('CTF / imaging')
    ctf.add_argument('--voltage', type=float, default=300.0,
                     help='Microscope voltage (kV)')
    ctf.add_argument('--amplitude-contrast', type=float, default=0.07)
    ctf.add_argument('--spherical-aberration', type=float, default=2.7,
                     help='Spherical aberration (mm)')
    ctf.add_argument('--calc-ctf', type=lambda x: x.lower() != 'false',
                     default=True, metavar='BOOL',
                     help='Enable per-tilt CTF weighting (True/False)')
    ctf.add_argument('--calc-exp', type=lambda x: x.lower() != 'false',
                     default=True, metavar='BOOL',
                     help='Enable exposure/dose weighting (True/False)')

    opt = p.add_argument_group('gapstop options')
    opt.add_argument('--binning', type=int, default=1,
                     help='Internal binning during template matching')
    opt.add_argument('--n-tiles', type=int, default=1,
                     help='Number of GPU tiles (-n flag to gapstop; set to 4 for 4 GPUs)')
    opt.add_argument('--scoring-fcn', default='flcf',
                     help='Scoring function (default: flcf)')
    opt.add_argument('--apply-laplacian', type=lambda x: x.lower() != 'false',
                     default=False, metavar='BOOL',
                     help='Apply Laplacian sharpening (True/False)')
    opt.add_argument('--noise-corr', type=lambda x: x.lower() != 'false',
                     default=True, metavar='BOOL',
                     help='Enable noise correlation correction (True/False)')
    opt.add_argument('--fourier-crop', type=lambda x: x.lower() != 'false',
                     default=True, metavar='BOOL',
                     help='Enable Fourier cropping (True/False)')
    opt.add_argument('--tiling', default='legacy_fix',
                     help='Tiling mode (default: legacy_fix)')
    opt.add_argument('--write-raw', type=lambda x: x.lower() != 'false',
                     default=False, metavar='BOOL',
                     help='Write raw score/angle maps in addition to MRC (True/False)')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report with central-slab tomogram and '
                         'score map side by side for each matched tomogram')
    qc.add_argument('--analyse-thickness', type=float, default=300.0, metavar='ANGST',
                    help='Slab thickness in Å for QC projections (default: 300 Å)')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/gapstop_match_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--output', '-o', default='gapstop_match',
                     help='Output directory (per-TS subdirectories are created inside)')
    ctl.add_argument('--gapstop-dir', default=None,
                     help='Directory containing the gapstop binary '
                          '(default: /opt/miniconda3/envs/gapstop/bin/)')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write STAR files and print commands without running gapstop')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    sep     = '─' * 70

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # Validate template and mask
    for path_arg, name in [(args.template, '--template'), (args.mask, '--mask')]:
        if not Path(path_arg).exists():
            print(f'ERROR: {name} {path_arg} not found')
            sys.exit(1)

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)

    # Locate gapstop binary
    gapstop_bin = _find_gapstop(args.gapstop_dir)
    if not gapstop_bin:
        msg = (f'gapstop not found.\n'
               f'  Expected at {_GAPSTOP_BIN}\n'
               f'  Or specify: --gapstop-dir /path/to/gapstop/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            gapstop_bin = 'gapstop'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    # Find volumes
    pairs = _find_volumes(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no tomogram volumes found in {in_dir}/')
        sys.exit(1)

    prefixes = [p for p, _ in pairs]
    vol_map  = {p: v for p, v in pairs}

    # include / exclude filtering
    if args.include:
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in inc)]
    if args.exclude:
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in exc)]

    # select-ts filter
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        orig_n   = len(prefixes)
        prefixes = [p for p in prefixes if p in selected_ts]
        n_excl   = orig_n - len(prefixes)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(prefixes)} remaining')

    if not prefixes:
        print('ERROR: no tomograms to process after filtering')
        sys.exit(1)

    print(f'Tomograms to process: {len(prefixes)}')
    print(sep)
    for p in prefixes[:10]:
        print(f'  {p}')
    if len(prefixes) > 10:
        print(f'  ... ({len(prefixes) - 10} more)')
    print(sep)

    do_qc      = getattr(args, 'analyse', False)
    qc_thick   = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    ok, failed = [], []

    for i, prefix in enumerate(prefixes):
        print(f'\n[{i+1}/{len(prefixes)}] {prefix}')

        tomo    = vol_map[prefix]
        ts_out  = out_dir / prefix
        tomo_id = i + 1  # 1-indexed integer ID for STOPGAP

        # Read metadata from AreTomo3 output
        try:
            tilt_angles, defocus_df, exposure_arr, frames = _read_ts_metadata(
                in_dir, prefix, dose_override=args.dose,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f'  ERROR: {exc}')
            failed.append(prefix)
            continue

        # Pixel size from MRC header (binned tomogram, not raw)
        angpix = _mrc_angpix(tomo)
        if angpix is None or angpix <= 0:
            print(f'  WARNING: could not read pixel size from {tomo.name}; using 1.0 Å')
            angpix = 1.0
        else:
            print(f'  Pixel size: {angpix:.4g} Å  (from MRC header)')

        nx, ny, nz = _mrc_dims(tomo)
        print(f'  Volume: {nx}×{ny}×{nz}  ({len(tilt_angles)} tilts)')

        if defocus_df is None:
            print(f'  CTF: no _CTF.txt found — wedge list will have no defocus')

        # Binary mask
        bmask = None
        if args.bmask_dir:
            bmask_path = Path(args.bmask_dir) / f'{prefix}{args.bmask_suffix}.mrc'
            if bmask_path.exists():
                bmask = bmask_path
                print(f'  Mask: {bmask}')
            else:
                print(f'  WARNING: bmask not found: {bmask_path}')

        # Prepare per-TS output directory
        ts_out.mkdir(parents=True, exist_ok=True)

        # Write wedge list STAR
        wedge_path = ts_out / f'{prefix}_wedgelist.star'
        _write_wedge_list(
            wedge_path, tomo_id, angpix, nx, ny, nz,
            tilt_angles, defocus_df, frames, exposure_arr, args,
        )
        print(f'  Wedge list: {wedge_path}')

        # Write gapstop parameter STAR
        param_path = ts_out / f'{prefix}_params.star'
        _write_gapstop_params(
            param_path, tomo, tomo_id, wedge_path,
            ts_out, args, bmask=bmask,
        )
        print(f'  Params:     {param_path}')

        # Build command
        cmd = [gapstop_bin, 'run_tm']
        if args.n_tiles > 1:
            cmd += ['-n', str(args.n_tiles)]
        cmd += [str(param_path)]

        _print_cmd(cmd)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(prefix)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: gapstop exited with code {ret.returncode}')
            failed.append(prefix)
            continue

        ok.append(prefix)

        # Report output maps
        for mrc in sorted(ts_out.glob('*.mrc')):
            if any(mrc.name.startswith(p) for p in ('scores_', 'angles_', 'noise_')):
                print(f'  → {mrc}')

        # QC
        if do_qc:
            before_b64 = after_b64 = None
            proj = central_slab_projection(tomo, qc_thick)
            if proj:
                before_b64 = projection_to_b64png(proj['img'])
            score_map = _find_score_map(ts_out)
            if score_map:
                proj_s = central_slab_projection(score_map, qc_thick)
                if proj_s:
                    after_b64 = projection_to_b64png(
                        proj_s['img'], pct=(50, 99.9), cmap='inferno', colorbar=True,
                    )
            qc_entries.append({
                'ts_name':    prefix,
                'before_b64': before_b64,
                'after_b64':  after_b64,
                'before_path': str(tomo),
                'after_path':  str(score_map) if score_map else '',
                'metadata': {
                    'pixel size':   f'{angpix:.4g} Å',
                    'volume':       f'{nx}×{ny}×{nz}',
                    'tilts':        str(len(tilt_angles)),
                    'n tiles':      str(args.n_tiles),
                    'lp / hp rad':  f'{args.lp_rad} / {args.hp_rad}',
                },
            })

    # Summary
    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')

    # QC report
    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'gapstop_match_qc.html')
        make_comparison_html(
            entries      = qc_entries,
            out_path     = html_path,
            title        = 'gapstop-match QC',
            command      = ' '.join(sys.argv),
            before_label = 'Tomogram',
            after_label  = 'Score map',
            slab_angst   = qc_thick,
        )
        print(f'QC report: {html_path}')

    if args.dry_run:
        return

    update_section(
        section='gapstop_match',
        values={
            'command':     ' '.join(sys.argv),
            'args':        args_to_dict(args),
            'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
            'n_processed': len(ok),
            'failed':      failed,
            'input_dir':   str(in_dir),
            'output_dir':  str(out_dir),
        },
        backup_dir=out_dir,
    )
