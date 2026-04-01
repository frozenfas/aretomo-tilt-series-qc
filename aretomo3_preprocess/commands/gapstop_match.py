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
import os
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
    slab_with_picks_b64, slab_picks_data,
    make_picks_html, make_picks_html_dev,
    make_comparison_html,
)

_GAPSTOP_BIN    = '/opt/miniconda3/envs/gapstop/bin/gapstop'
_GAPSTOP_PYTHON = '/opt/miniconda3/envs/gapstop/bin/python'


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


def _find_gapstop_python(gapstop_dir=None):
    """Return path to gapstop env python, or None if not found."""
    candidates = []
    if gapstop_dir:
        candidates.append(str(Path(gapstop_dir) / 'python'))
    candidates.append(_GAPSTOP_PYTHON)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('python3')


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
    gapstop CTF filter requires a single 'defocus' column (µm); we use the
    mean of defocus1 and defocus2. Phase shift included if non-zero.
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
        defocus_um, pshift = [], []
        for f in frames:
            entry = defocus_df.get(f['sec'])
            if entry:
                # mean of defocus1 and defocus2; gapstop requires single 'defocus' column
                d_um = (entry['defocus1_A'] + entry['defocus2_A']) / 2.0 / 1e4
                defocus_um.append(d_um)
                pshift.append(entry['phase_shift_rad'])
            else:
                defocus_um.append(float('nan'))
                pshift.append(0.0)
        data['defocus'] = defocus_um
        # Only write pshift if any value is non-zero (phase plate data).
        # gapstop bug: _array() returns a list so pshift[:,None] crashes;
        # omitting the column lets gapstop fall back to np.zeros_like(defocus).
        if any(p != 0.0 for p in pshift):
            data['pshift'] = pshift

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
        'tomo_mask_name': [str(bmask) if bmask else None],
        'wedgelist_name': [str(wedge_path)],
        'tmpl_name':      [str(Path(args.template).resolve())],
        'mask_name':      [str(Path(args.mask).resolve())],
        'symmetry':       [args.symmetry],
        'anglist_name':   [None],
        'anglist_order':  [args.anglist_order],
        'angincr':        [args.angincr],
        'angiter':        [args.angiter],
        'phi_angincr':    [args.phi_angincr],
        'phi_angiter':    [args.phi_angiter],
        'tilelist_name':  [None],
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
    # Drop optional keys whose value is None — gapstop treats missing columns as absent
    params = {k: v for k, v in params.items() if v != [None]}
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


def _find_em_maps(ts_out):
    """Return (scores_map, angles_map) paths, or (None, None) if not found.
    gapstop writes .mrc (newer) or .em (older) depending on version."""
    for ext in ('mrc', 'em'):
        scores = sorted(ts_out.glob(f'scores_*.{ext}'))
        angles = sorted(ts_out.glob(f'angles_*.{ext}'))
        if scores and angles:
            return scores[0], angles[0]
    return None, None


def _angles_list_name(angincr, angiter, phi_angincr, phi_angiter):
    """Return the filename gapstop auto-generates for the angles list."""
    return f'angles_{angincr}_{angiter}_{phi_angincr}_{phi_angiter}.txt'


def _find_angles_list(angincr, angiter, phi_angincr, phi_angiter, search_dirs):
    """Search directories for the gapstop-generated angles list text file."""
    name = _angles_list_name(angincr, angiter, phi_angincr, phi_angiter)
    for d in search_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def _read_params_star(param_path):
    """Read tomo_num and angular params from a per-TS params STAR file."""
    try:
        import starfile
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = starfile.read(str(param_path), always_dict=True)
        df = next(iter(data.values()))
        row = df.iloc[0]
        return {
            'tomo_num':    int(row['tomo_num']),
            'angincr':     float(row['angincr']),
            'angiter':     float(row['angiter']),
            'phi_angincr': float(row['phi_angincr']),
            'phi_angiter': float(row['phi_angiter']),
            'symmetry':    str(row['symmetry']),
            'anglist_order': str(row.get('anglist_order', 'zxz')),
            'rootdir':     str(row.get('rootdir', '.')),
        }
    except Exception as exc:
        return None


def _build_extract_script(scores_em, angles_em, angles_list, tomo_id,
                           diam_px, threshold, n_particles, out_star, symmetry,
                           anglist_order):
    """Return a Python script string that runs cryoCAT extraction."""
    n_part_arg = f'n_particles={n_particles},' if n_particles is not None else ''
    return (
        'from cryocat import tmana, cryomap; '
        f'scores = cryomap.read("{scores_em}"); '
        f'angles = cryomap.read("{angles_em}"); '
        f'tmana.scores_extract_particles('
        f'scores_map=scores, angles_map=angles, '
        f'angles_list="{angles_list}", '
        f'tomo_id={tomo_id}, '
        f'particle_diameter={diam_px}, '
        f'scores_threshold={threshold}, '
        f'{n_part_arg}'
        f'output_path="{out_star}", '
        f'output_type="relion", '
        f'angles_order="{anglist_order}", '
        f'symmetry="{symmetry}", '
        f'angles_numbering=0)'
    )


def _run_extraction(ts_out, tomo_id, angpix, angles_list, python_bin, args,
                    dry_run=False):
    """Run cryoCAT extraction on a completed gapstop output directory."""
    scores_em, angles_em = _find_em_maps(ts_out)
    if scores_em is None or angles_em is None:
        print('  WARNING: scores_*.em or angles_*.em not found — skipping extraction')
        return False
    if angles_list is None:
        print('  WARNING: angles list file not found — skipping extraction')
        return False

    diam_px = int(round(args.particle_diameter / angpix)) if args.particle_diameter else 20
    n_particles = getattr(args, 'n_particles', None)
    threshold   = args.scores_threshold
    symmetry    = getattr(args, 'symmetry', 'c1').lower()
    anglist_order = getattr(args, 'anglist_order', 'zxz')
    out_star = ts_out / f'{ts_out.name}_particles.star'

    script = _build_extract_script(
        scores_em, angles_em, angles_list, tomo_id,
        diam_px, threshold, n_particles, out_star,
        symmetry, anglist_order,
    )
    cmd = [python_bin, '-c', script]

    # Print in readable form
    print(f'  Extraction:')
    print(f'    scores:    {scores_em.name}')
    print(f'    angles:    {angles_em.name}')
    print(f'    threshold: {threshold}')
    print(f'    diam:      {diam_px} px  ({args.particle_diameter} Å / {angpix:.4g} Å/px)'
          if args.particle_diameter else f'    diam:      {diam_px} px')
    print(f'    → {out_star}')

    if dry_run:
        print('  [dry-run: skipping extraction]')
        return True

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f'  ERROR: extraction failed (exit {ret.returncode})')
        return False

    if out_star.exists():
        import subprocess as _sp
        n = int(_sp.run(['wc', '-l', str(out_star)],
                        capture_output=True, text=True).stdout.split()[0])
        print(f'  → {out_star}  (~{max(0,n-10)} particles)')
    return True


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
    inp.add_argument('--input', '-i', default=None,
                     help='AreTomo3 output directory containing ts-*_Vol.mrc files '
                          '(not required with --extract-only)')
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

    tmpl = p.add_argument_group('template matching (required unless --extract-only)')
    tmpl.add_argument('--template', '-t', default=None,
                      help='Template MRC file')
    tmpl.add_argument('--mask', '-m', default=None,
                      help='Mask MRC file')

    ang = p.add_argument_group('angular search')
    ang.add_argument('--angincr', type=float, default=None,
                     help='Angular increment (degrees) for the primary search axis')
    ang.add_argument('--angiter', type=float, default=None,
                     help='Angular iterations for the primary axis '
                          '(search range ≈ angincr × angiter)')
    ang.add_argument('--phi-angincr', type=float, default=None,
                     help='Angular increment (degrees) for the phi (azimuth) axis')
    ang.add_argument('--phi-angiter', type=float, default=None,
                     help='Angular iterations for the phi axis')
    ang.add_argument('--anglist-order', default='zxz',
                     help='Euler angle convention for angular search (default: zxz)')
    ang.add_argument('--symmetry', default='C1',
                     help='Particle symmetry (e.g. C1, C2, C6)')

    flt = p.add_argument_group('filter parameters')
    flt.add_argument('--lp-rad', type=float, default=None,
                     help='Low-pass filter radius (Nyquist fraction, e.g. 0.35)')
    flt.add_argument('--lp-sigma', type=float, default=3.0,
                     help='Low-pass filter sigma (Nyquist fraction)')
    flt.add_argument('--hp-rad', type=float, default=None,
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
    opt.add_argument('--gpu', type=int, nargs='+', default=None,
                     metavar='ID',
                     help='GPU device IDs to use (sets CUDA_VISIBLE_DEVICES). '
                          'E.g. --gpu 0 1 uses GPUs 0 and 1. '
                          'Also sets --n-tiles to the number of GPUs if not specified.')
    opt.add_argument('--n-tiles', type=int, default=None,
                     help='Number of GPU tiles (-n flag to gapstop). '
                          'Defaults to the number of --gpu IDs, or 1 if --gpu not set.')
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

    ext = p.add_argument_group('extraction (runs immediately after each TS is matched)')
    ext.add_argument('--extract', action='store_true',
                     help='Run cryoCAT extraction after each tomogram is matched')
    ext.add_argument('--scores-threshold', type=float, default=0.08,
                     help='Score threshold for particle extraction')
    ext.add_argument('--n-particles', type=int, default=None,
                     help='Maximum number of particles to extract per tomogram '
                          '(default: no limit)')
    ext.add_argument('--particle-diameter', type=float, default=None, metavar='ANGST',
                     help='Particle diameter in Å — used to set minimum peak spacing '
                          'during extraction (e.g. 280 for 80S ribosome)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--output', '-o', default='gapstop_match',
                     help='Output directory (per-TS subdirectories are created inside)')
    ctl.add_argument('--extract-only', action='store_true',
                     help='Skip template matching; run extraction on existing score maps '
                          'in --output.  --input is not required in this mode.')
    ctl.add_argument('--gapstop-dir', default=None,
                     help='Directory containing the gapstop binary '
                          '(default: /opt/miniconda3/envs/gapstop/bin/)')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write STAR files and print commands without running gapstop')

    p.set_defaults(func=run)
    return p


def _picks_qc_entry(prefix, tomo_path, star_path, angpix, score_map, args):
    """Build a QC entry dict with tomogram+picks overlay and score map."""
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    diam     = getattr(args, 'particle_diameter', None)

    img_b64 = score_b64 = None

    # Tomogram slab with picks overlay
    picks_data = None
    try:
        import warnings
        import starfile
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = starfile.read(str(star_path), always_dict=True)
        df = next(iter(data.values()))
        result = slab_with_picks_b64(tomo_path, df,
                                     slab_angst=qc_thick,
                                     particle_diameter=diam)
        if result:
            img_b64  = result['img_b64']
            n_total  = result['n_total']
            n_shown  = result['n_shown']
        else:
            n_total = n_shown = 0
        picks_data = slab_picks_data(tomo_path, df,
                                     slab_angst=qc_thick,
                                     particle_diameter=diam)
    except Exception:
        n_total = n_shown = 0

    # Score map
    if score_map and score_map.exists():
        proj = central_slab_projection(score_map, qc_thick)
        if proj:
            score_b64 = projection_to_b64png(
                proj['img'], pct=(50, 99.9), cmap='inferno', colorbar=True,
            )

    return {
        'ts_name':   prefix,
        'img_b64':   img_b64,
        'score_b64': score_b64,
        'picks_data': picks_data,
        'n_total':   n_total,
        'n_shown':   n_shown,
        'tomo_path': str(tomo_path),
        'metadata': {
            'pixel size':  f'{angpix:.4g} Å',
            'threshold':   str(args.scores_threshold),
            'particles':   f'{n_shown} in slab / {n_total} total',
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extract-only mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_extract_only(args, out_dir, sep):  # noqa: C901
    python_bin = _find_gapstop_python(args.gapstop_dir)
    if not python_bin:
        print(f'ERROR: gapstop Python not found at {_GAPSTOP_PYTHON}')
        sys.exit(1)

    # Find per-TS subdirectories that have a params STAR and score maps
    ts_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    if not ts_dirs:
        print(f'ERROR: no per-TS subdirectories found in {out_dir}')
        sys.exit(1)

    # select-ts filter
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))

    jobs = []
    for ts_dir in ts_dirs:
        ts_name = ts_dir.name
        if selected_ts is not None and ts_name not in selected_ts:
            continue
        param_file = ts_dir / f'{ts_name}_params.star'
        if not param_file.exists():
            continue
        scores_em, angles_em = _find_em_maps(ts_dir)
        if scores_em is None:
            continue
        jobs.append((ts_name, ts_dir, param_file))

    if not jobs:
        print(f'ERROR: no completed gapstop runs found in {out_dir}')
        sys.exit(1)

    print(f'Tomograms to extract: {len(jobs)}')
    print(sep)
    for ts_name, _, _ in jobs[:10]:
        print(f'  {ts_name}')
    if len(jobs) > 10:
        print(f'  ... ({len(jobs) - 10} more)')
    print(sep)

    do_qc      = getattr(args, 'analyse', False)
    qc_entries = []
    ok, failed = [], []

    for ts_name, ts_dir, param_file in jobs:
        print(f'\n{ts_name}')

        params = _read_params_star(param_file)
        if params is None:
            print(f'  ERROR: could not read {param_file.name}')
            failed.append(ts_name)
            continue

        tomo_id   = params['tomo_num']
        tomo_path = Path('.')

        # Pixel size from tomogram MRC (path stored in params STAR as tomo_name)
        try:
            import starfile, warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = starfile.read(str(param_file), always_dict=True)
            tomo_path = Path(next(iter(data.values())).iloc[0]['tomo_name'])
            angpix = _mrc_angpix(tomo_path) or 1.0
        except Exception:
            angpix = 1.0

        # Find angles list — check rootdir from params, then cwd, then out_dir
        angles_list = _find_angles_list(
            params['angincr'], params['angiter'],
            params['phi_angincr'], params['phi_angiter'],
            search_dirs=[params['rootdir'], Path('.'), out_dir],
        )

        ok_ts = _run_extraction(
            ts_dir, tomo_id, angpix, angles_list, python_bin, args,
            dry_run=args.dry_run,
        )
        (ok if ok_ts else failed).append(ts_name)

        if ok_ts and do_qc and not args.dry_run:
            star_path = ts_dir / f'{ts_name}_particles.star'
            score_map = _find_score_map(ts_dir)
            if star_path.exists():
                qc_entries.append(_picks_qc_entry(
                    ts_name, tomo_path, star_path, angpix, score_map, args,
                ))

    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')

    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'gapstop_extract_qc.html')
        make_picks_html(
            entries   = qc_entries,
            out_path  = html_path,
            title     = 'gapstop-match extraction QC',
            command   = ' '.join(sys.argv),
            slab_angst = getattr(args, 'analyse_thickness', 300.0),
        )
        make_picks_html_dev(
            entries   = qc_entries,
            out_path  = html_path.with_stem(html_path.stem + '_dev'),
            title     = 'gapstop-match extraction QC',
            command   = ' '.join(sys.argv),
            slab_angst = getattr(args, 'analyse_thickness', 300.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output).resolve()
    sep     = '─' * 70

    extract_only = getattr(args, 'extract_only', False)
    if extract_only:
        _run_extract_only(args, out_dir, sep)
        return

    if args.input is None:
        print('ERROR: --input is required unless --extract-only is set')
        sys.exit(1)

    in_dir = Path(args.input).resolve()

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # Validate required TM args
    missing = [name for val, name in [
        (args.template,    '--template'),
        (args.mask,        '--mask'),
        (args.angincr,     '--angincr'),
        (args.angiter,     '--angiter'),
        (args.phi_angincr, '--phi-angincr'),
        (args.phi_angiter, '--phi-angiter'),
        (args.lp_rad,      '--lp-rad'),
        (args.hp_rad,      '--hp-rad'),
    ] if val is None]
    if missing:
        print(f'ERROR: the following arguments are required for template matching: {", ".join(missing)}')
        sys.exit(1)

    # Validate template and mask paths
    for path_arg, name in [(args.template, '--template'), (args.mask, '--mask')]:
        if not Path(path_arg).exists():
            print(f'ERROR: {name} {path_arg} not found')
            sys.exit(1)

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)

    # Locate gapstop binary and python
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

    # Resolve n_tiles and CUDA_VISIBLE_DEVICES
    n_tiles = args.n_tiles if args.n_tiles is not None else (len(args.gpu) if args.gpu else 1)
    cuda_env = os.environ.copy()
    if args.gpu:
        cuda_env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in args.gpu)
        print(f'  GPUs: {args.gpu}  (CUDA_VISIBLE_DEVICES={cuda_env["CUDA_VISIBLE_DEVICES"]})')

    do_extract = getattr(args, 'extract', False)
    python_bin = None
    if do_extract:
        python_bin = _find_gapstop_python(args.gapstop_dir)
        if not python_bin:
            print(f'WARNING: gapstop Python not found at {_GAPSTOP_PYTHON} '
                  f'— --extract will be skipped')
            do_extract = False

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
        if n_tiles > 1:
            cmd += ['-n', str(n_tiles)]
        cmd += [str(param_path)]

        _print_cmd(cmd)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            if do_extract:
                angles_list = _find_angles_list(
                    args.angincr, args.angiter, args.phi_angincr, args.phi_angiter,
                    search_dirs=[Path('.'), out_dir, ts_out],
                )
                _run_extraction(ts_out, tomo_id, angpix, angles_list,
                                python_bin, args, dry_run=True)
            ok.append(prefix)
            continue

        ret = subprocess.run(cmd, env=cuda_env)
        if ret.returncode != 0:
            print(f'  ERROR: gapstop exited with code {ret.returncode}')
            failed.append(prefix)
            continue

        ok.append(prefix)

        # Report output maps
        for mrc in sorted(ts_out.glob('*.mrc')):
            if any(mrc.name.startswith(p) for p in ('scores_', 'angles_', 'noise_')):
                print(f'  → {mrc}')

        # Extraction
        if do_extract:
            angles_list = _find_angles_list(
                args.angincr, args.angiter, args.phi_angincr, args.phi_angiter,
                search_dirs=[Path('.'), out_dir, ts_out],
            )
            _run_extraction(ts_out, tomo_id, angpix, angles_list,
                            python_bin, args, dry_run=False)

        # QC
        if do_qc:
            score_map = _find_score_map(ts_out)
            star_path = ts_out / f'{prefix}_particles.star'

            if do_extract and star_path.exists():
                # Show tomogram with picks overlay + score map
                qc_entries.append(_picks_qc_entry(
                    prefix, tomo, star_path, angpix, score_map, args,
                ))
            else:
                # Matching only — show tomogram slab vs score map
                before_b64 = after_b64 = None
                proj = central_slab_projection(tomo, qc_thick)
                if proj:
                    before_b64 = projection_to_b64png(proj['img'])
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
                        'pixel size':  f'{angpix:.4g} Å',
                        'volume':      f'{nx}×{ny}×{nz}',
                        'tilts':       str(len(tilt_angles)),
                        'n tiles':     str(n_tiles),
                        'lp / hp rad': f'{args.lp_rad} / {args.hp_rad}',
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
        if do_extract:
            make_picks_html(
                entries   = qc_entries,
                out_path  = html_path,
                title     = 'gapstop-match QC',
                command   = ' '.join(sys.argv),
                slab_angst = qc_thick,
            )
            make_picks_html_dev(
                entries   = qc_entries,
                out_path  = html_path.with_stem(html_path.stem + '_dev'),
                title     = 'gapstop-match QC',
                command   = ' '.join(sys.argv),
                slab_angst = qc_thick,
            )
        else:
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
