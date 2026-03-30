"""
pytom-match — batch template matching with pytom-match-pick on AreTomo3 tomograms.

Adapted from batch_pytom_aretomo3 by Phaips
(https://github.com/Phaips/batch_pytom_aretomo3, MIT licence).
Key differences from the original script:
  - Uses shared parsers (parse_aln_file, parse_ctf_file, parse_tlt_file)
    rather than custom readers; acquisition order is read from _TLT.txt
    instead of the IMOD _order_list.csv
  - Runs pytom_match_template.py directly (no SLURM) using the pytom_tm
    conda environment at /opt/miniconda3/envs/pytom_tm/ (default)
  - --select-ts integration for TS filtering
  - project.json bookkeeping

pytom-match-pick must be installed in the pytom_tm conda environment at
/opt/miniconda3/envs/pytom_tm/ (default), or be available on PATH.

Typical usage
-------------
  # Run on all tomograms in run001/
  aretomo3-preprocess pytom-match \\
      --input run001 \\
      --template ribosome_14A.mrc \\
      --mask ribosome_mask.mrc \\
      --voxel-size 14.0 \\
      --gpu 0 1 \\
      --particle-diameter 300 \\
      --output pytom_match \\
      --dry-run

  # Run on selected tomograms only
  aretomo3-preprocess pytom-match \\
      --input run001 \\
      --template ribosome_14A.mrc --mask ribosome_mask.mrc \\
      --voxel-size 14.0 --gpu 0 1 \\
      --particle-diameter 300 \\
      --select-ts run001_analysis/ts-select.csv \\
      --output pytom_match

Optional: run candidate extraction immediately after each tomogram is matched
so results can be inspected in 3dmod while the remaining TS are still running:

  aretomo3-preprocess pytom-match \\
      --input run001 ... \\
      --extract --n-particles 2000 --tophat-filter --relion5-compat --imod
"""

import json
import re
import shutil
import struct
import sys
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file,
)
from aretomo3_preprocess.shared.project_json import (
    update_section, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import resolve_selected_ts

# Default pytom binary location
_PYTOM_BIN = '/opt/miniconda3/envs/pytom_tm/bin/pytom_match_template.py'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_pytom(pytom_dir=None):
    """Return path to pytom_match_template.py, or None if not found."""
    candidates = []
    if pytom_dir:
        candidates.append(str(Path(pytom_dir) / 'pytom_match_template.py'))
    candidates.append(_PYTOM_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('pytom_match_template.py')


_PYTOM_EXTRACT_BIN = '/opt/miniconda3/envs/pytom_tm/bin/pytom_extract_candidates.py'


def _find_pytom_extract(pytom_dir=None):
    """Return path to pytom_extract_candidates.py, or None if not found."""
    candidates = []
    if pytom_dir:
        candidates.append(str(Path(pytom_dir) / 'pytom_extract_candidates.py'))
    candidates.append(_PYTOM_EXTRACT_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('pytom_extract_candidates.py')


def _mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC header without mrcfile dependency."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)


def _find_job_jsons(input_dir, selected_ts=None):
    """Return sorted list of (ts_name, job_json_path) tuples."""
    jobs = []
    for ts_dir in sorted(Path(input_dir).iterdir()):
        if not ts_dir.is_dir():
            continue
        ts_name = ts_dir.name
        if selected_ts is not None and ts_name not in selected_ts:
            continue
        matches = sorted(ts_dir.glob('*_job.json'))
        if matches:
            jobs.append((ts_name, matches[0]))
    return jobs


def _star_to_mod(star_path, job_json_path, mod_dir):
    """
    Convert a RELION5 particles STAR file to an IMOD .mod point model.

    Adapted from rln2mod by Phaips (https://github.com/Phaips/rln2mod).
    """
    try:
        import starfile
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
    except ImportError:
        print('  WARNING: starfile not installed — skipping IMOD conversion')
        print('           conda run -n pytom_tm pip install starfile')
        return False

    with open(job_json_path) as fh:
        job = json.load(fh)
    tomo_path = job.get('tomogram') or job.get('tomogram_path') or job.get('volume_path')
    if not tomo_path or not Path(tomo_path).exists():
        print(f'  WARNING: cannot find tomogram path in {job_json_path.name} — skipping IMOD conversion')
        return False

    nx, ny, nz = _mrc_dims(tomo_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        data = starfile.read(str(star_path))
    df = data[next(iter(data))] if isinstance(data, dict) else data

    required = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
                 'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    missing = required - set(df.columns)
    if missing:
        print(f'  WARNING: STAR file missing columns {missing} — '
              f're-run extraction with --relion5-compat')
        return False

    px = df['rlnTomoTiltSeriesPixelSize']
    xs = df['rlnCenteredCoordinateXAngst'] / px + nx / 2
    ys = df['rlnCenteredCoordinateYAngst'] / px + ny / 2
    zs = df['rlnCenteredCoordinateZAngst'] / px + nz / 2

    mod_dir.mkdir(parents=True, exist_ok=True)
    stem = star_path.stem
    txt  = mod_dir / f'{stem}.txt'
    mod  = mod_dir / f'{stem}.mod'

    with open(txt, 'w') as fh:
        for x, y, z in zip(xs, ys, zs):
            fh.write(f'{x:.6f} {y:.6f} {z:.6f}\n')

    point2model = shutil.which('point2model')
    if not point2model:
        print('  WARNING: point2model not found on PATH — load IMOD first')
        txt.unlink(missing_ok=True)
        return False

    ret = subprocess.run([point2model, str(txt), str(mod)], capture_output=True)
    if ret.returncode != 0:
        print(f'  WARNING: point2model failed: {ret.stderr.decode().strip()}')
        return False

    txt.unlink(missing_ok=True)
    print(f'  → {mod}  ({len(df)} particles)')
    return True


def _find_tomogram(aretomo_dir, prefix, vol_suffix):
    """Locate the reconstructed volume for a given prefix.

    AreTomo3 naming conventions:
      ts-xxx_Vol.mrc       (single --at-bin run, default)
      ts-xxx_b4_Vol.mrc    (multi-bin, vol_suffix='_b4')
      ts-xxx.mrc           (fallback if no _Vol suffix)
    """
    aretomo_dir = Path(aretomo_dir)
    candidates = []
    if vol_suffix:
        candidates.append(aretomo_dir / f'{prefix}{vol_suffix}_Vol.mrc')
        candidates.append(aretomo_dir / f'{prefix}{vol_suffix}.mrc')
    candidates.append(aretomo_dir / f'{prefix}_Vol.mrc')
    candidates.append(aretomo_dir / f'{prefix}.mrc')
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_ts_metadata(aretomo_dir, prefix, dose_override=None):
    """
    Read tilt angles, defocus, and cumulative prior dose for one TS prefix.

    Adapted from batch_pytom_aretomo3.py (Phaips/batch_pytom_aretomo3).
    The original read from _Imod/_st.tlt + _CTF.txt + _order_list.csv; here
    we use shared parsers on .aln + _CTF.txt + _TLT.txt instead.

    Returns
    -------
    tlt_out      : list of tilt angles (tilt-sorted, dark frames excluded)
    defocus_out  : list of mean defocus (µm), or None if _CTF.txt absent
    exposure_out : list of cumulative prior dose (e⁻/Å²), RELION convention
                   (first acquired frame = 0, second = dose[1], etc.)
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

    # Aligned frames in tilt-sorted order; dark frames excluded by parse_aln_file
    frames = aln['frames']
    if not frames:
        raise ValueError(f'{prefix}: no aligned frames in .aln')

    # Tilt angles: use corrected tilts from .aln (same as IMOD _st.tlt)
    tlt_out = [f['tilt'] for f in frames]

    # Defocus from _CTF.txt (optional)
    defocus_out = None
    if ctf_path.exists():
        ctf = parse_ctf_file(ctf_path)
        defocus_out = []
        for f in frames:
            sec = f['sec']
            if sec not in ctf:
                raise ValueError(f'{prefix}: no CTF entry for sec {sec}')
            defocus_out.append(ctf[sec]['mean_defocus_um'])

    # Cumulative prior dose (RELION convention: prior to each acquisition)
    # Sort all rows by acq_order, accumulate dose, then index back by tilt order.
    # dose_override (CLI --dose) replaces per-frame dose from _TLT.txt.
    sorted_rows = sorted(tlt.values(), key=lambda r: r['acq_order'])
    cum = 0.0
    acq_to_prior = {}
    for row in sorted_rows:
        acq_to_prior[row['acq_order']] = round(cum, 2)
        per_frame = dose_override if dose_override is not None else row['dose_e_per_A2']
        cum += per_frame

    exposure_out = []
    for f in frames:
        sec = f['sec']
        if sec not in tlt:
            raise ValueError(f'{prefix}: no _TLT.txt entry for sec {sec}')
        exposure_out.append(acq_to_prior[tlt[sec]['acq_order']])

    return tlt_out, defocus_out, exposure_out


def _write_aux_files(out_dir, prefix, tlt_out, defocus_out, exposure_out):
    """Write per-tomogram .tlt, _defocus.txt, and _exposure.txt files."""
    od = Path(out_dir) / prefix
    od.mkdir(parents=True, exist_ok=True)

    tlt_file = od / f'{prefix}.tlt'
    exp_file = od / f'{prefix}_exposure.txt'
    df_file  = od / f'{prefix}_defocus.txt' if defocus_out is not None else None

    tlt_file.write_text(''.join(f'{t}\n' for t in tlt_out))
    exp_file.write_text(''.join(f'{e}\n' for e in exposure_out))
    if defocus_out is not None:
        df_file.write_text(''.join(f'{d}\n' for d in defocus_out))

    return tlt_file, df_file, exp_file


def _build_cmd(pytom_bin, tomo, tlt_file, df_file, exp_file, out_subdir, args, bmask=None):
    """
    Build the pytom_match_template.py command for one tomogram.

    Argument order mirrors batch_pytom_aretomo3.py (Phaips/batch_pytom_aretomo3)
    with SLURM-specific parts removed.
    """
    cmd = [
        pytom_bin,
        '-v', str(tomo),
        '-a', str(tlt_file),
        '--dose-accumulation', str(exp_file),
        '-t', str(args.template),
        '-d', str(out_subdir),
        '-m', str(args.mask),
    ]

    if df_file is not None:
        cmd += ['--defocus', str(df_file)]

    if args.particle_diameter:
        cmd += ['--particle-diameter', str(args.particle_diameter)]
    if args.angular_search:
        cmd += ['--angular-search', str(args.angular_search)]
    if args.voxel_size:
        cmd += ['--voxel-size-angstrom', str(args.voxel_size)]
    if args.low_pass:
        cmd += ['--low-pass', str(args.low_pass)]
    if args.high_pass:
        cmd += ['--high-pass', str(args.high_pass)]
    if args.random_phase_correction:
        cmd += ['-r', '--rng-seed', str(args.rng_seed)]
    cmd += ['-g'] + [str(g) for g in args.gpu]
    if args.per_tilt_weighting:
        cmd += ['--per-tilt-weighting']
    if df_file is not None and args.tomogram_ctf_model:
        cmd += ['--tomogram-ctf-model', args.tomogram_ctf_model]
    if args.non_spherical_mask:
        cmd += ['--non-spherical-mask']
    if args.spectral_whitening:
        cmd += ['--spectral-whitening']
    if args.half_precision:
        cmd += ['--half-precision']
    if args.defocus_handedness is not None:
        cmd += ['--defocus-handedness', str(args.defocus_handedness)]
    if bmask is not None:
        cmd += ['--tomogram-mask', str(bmask)]
    if args.phase_shift is not None:
        cmd += ['--phase-shift', str(args.phase_shift)]
    if args.z_axis_rotational_symmetry is not None:
        cmd += ['--z-axis-rotational-symmetry', str(args.z_axis_rotational_symmetry)]
    if args.volume_split:
        cmd += ['-s'] + [str(x) for x in args.volume_split]
    if args.search_x:
        cmd += ['--search-x'] + [str(x) for x in args.search_x]
    if args.search_y:
        cmd += ['--search-y'] + [str(y) for y in args.search_y]
    if args.search_z:
        cmd += ['--search-z'] + [str(z) for z in args.search_z]
    if args.log:
        cmd += ['--log', args.log]
    cmd += [
        '--amplitude-contrast', str(args.amplitude_contrast),
        '--spherical-aberration', str(args.spherical_aberration),
        '--voltage', str(args.voltage),
    ]

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Parser registration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'pytom-match',
        help='Batch template matching with pytom-match-pick on AreTomo3 tomograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', default=None,
                     help='AreTomo3 output directory (not required with --extract-only)')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume file suffix, e.g. "_b4" for ts-xxx_b4_Vol.mrc; '
                          'leave empty to auto-detect (tries _Vol.mrc first)')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv from select-ts; only selected TS are processed')
    inp.add_argument('--include', nargs='+',
                     help='Process only these TS prefixes (wildcards supported)')
    inp.add_argument('--exclude', nargs='+',
                     help='Exclude these TS prefixes (wildcards supported)')
    inp.add_argument('--bmask-dir', default=None,
                     help='Directory of per-TS boundary mask MRCs (<prefix>.mrc)')
    inp.add_argument('--dose', type=float, default=None,
                     help='Per-frame dose (e⁻/Å²); if omitted, reads from _TLT.txt')

    tmpl = p.add_argument_group('template matching (required)')
    tmpl.add_argument('--template', '-t', required=True,
                      help='Template MRC file (inverted contrast for cryo-ET)')
    tmpl.add_argument('--mask', '-m', required=True,
                      help='Mask MRC file')
    tmpl.add_argument('--voxel-size', type=float, required=True,
                      help='Voxel size in Å (tomogram and template must match)')
    tmpl.add_argument('--gpu', '-g', nargs='+', type=int, required=True,
                      help='GPU ID(s) for pytom_match_template.py')

    ang = p.add_argument_group('angular search (exactly one required)')
    ang_grp = ang.add_mutually_exclusive_group(required=True)
    ang_grp.add_argument('--particle-diameter', type=float,
                         help='Particle diameter in Å (Crowther criterion sampling)')
    ang_grp.add_argument('--angular-search',
                         help='Angular search step in degrees, or path to .txt rotation list')

    opt = p.add_argument_group('pytom optional')
    opt.add_argument('--non-spherical-mask', action='store_true',
                     help='Enable non-spherical mask')
    opt.add_argument('--z-axis-rotational-symmetry', type=int, default=None,
                     help='Z-axis rotational symmetry fold')
    opt.add_argument('--volume-split', nargs=3, type=int, metavar=('X', 'Y', 'Z'),
                     help='Split volume into X Y Z blocks (reduces GPU memory)')
    opt.add_argument('--search-x', nargs=2, type=int, metavar=('START', 'END'),
                     help='Restrict search range along X')
    opt.add_argument('--search-y', nargs=2, type=int, metavar=('START', 'END'),
                     help='Restrict search range along Y')
    opt.add_argument('--search-z', nargs=2, type=int, metavar=('START', 'END'),
                     help='Restrict search range along Z')
    opt.add_argument('--tomogram-ctf-model', choices=['phase-flip'],
                     help='CTF model used in reconstruction (only with --defocus)')
    opt.add_argument('-r', '--random-phase-correction', action='store_true',
                     help='Enable random-phase correction (STOPGAP-style)')
    opt.add_argument('--rng-seed', type=int, default=69,
                     help='RNG seed for random-phase correction')
    opt.add_argument('--half-precision', action='store_true',
                     help='Use float16 output')
    opt.add_argument('--per-tilt-weighting', action='store_true',
                     help='Enable per-tilt CTF weighting')
    opt.add_argument('--low-pass', type=float, default=None,
                     help='Low-pass filter cutoff (Å)')
    opt.add_argument('--high-pass', type=float, default=None,
                     help='High-pass filter cutoff (Å)')
    opt.add_argument('--spectral-whitening', action='store_true',
                     help='Enable spectral whitening')
    opt.add_argument('--phase-shift', type=float, default=None,
                     help='Phase shift in degrees')
    opt.add_argument('--defocus-handedness', type=int, choices=[-1, 0, 1], default=None,
                     help='Defocus gradient handedness')
    opt.add_argument('--log', choices=['info', 'debug'], default=None,
                     help='Logging level for pytom_match_template.py')

    ctf = p.add_argument_group('CTF / imaging (passed unconditionally)')
    ctf.add_argument('--amplitude-contrast', type=float, default=0.07)
    ctf.add_argument('--spherical-aberration', type=float, default=2.7,
                     help='Spherical aberration (mm)')
    ctf.add_argument('--voltage', type=float, default=300,
                     help='Voltage (kV)')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report with central-slab tomogram and '
                         'score map side by side for each matched tomogram')
    qc.add_argument('--analyse-thickness', type=float, default=300.0, metavar='ANGST',
                    help='Slab thickness in Å for QC projections (default: 300 Å)')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/pytom_match_qc.html)')

    ext = p.add_argument_group('extraction (runs immediately after each TS is matched)')
    ext.add_argument('--extract', action='store_true',
                     help='Run pytom_extract_candidates.py after each tomogram is matched '
                          'so picks are available while remaining TS are still running')
    ext.add_argument('--n-particles', type=int, default=None,
                     help='Max candidates to extract per tomogram (required with --extract)')
    ext.add_argument('--tophat-filter', action='store_true',
                     help='Apply tophat background flattening before extraction '
                          '(recommended)')
    ext.add_argument('--tophat-bins', type=int, default=None,
                     help='Number of bins for tophat filter')
    ext.add_argument('--cut-off', type=float, default=None,
                     help='Override automated LCCmax cut-off')
    ext.add_argument('--n-false-positives', type=int, default=None,
                     help='False positives for cut-off estimation (default 1)')
    ext.add_argument('--relion5-compat', action='store_true',
                     help='Write RELION5-compatible STAR files (required for --imod)')
    ext.add_argument('--imod', action='store_true',
                     help='Convert extracted STAR files to IMOD .mod point models '
                          'for 3dmod visualisation (requires --relion5-compat). '
                          'Adapted from rln2mod (https://github.com/Phaips/rln2mod).')
    ext.add_argument('--imod-dir', default=None,
                     help='Directory for .mod files (default: <output>/mod/)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--output', '-o', default='pytom_match',
                     help='Output directory (per-TS subdirectories are created inside)')
    ctl.add_argument('--extract-only', action='store_true',
                     help='Skip template matching; run extraction on existing job JSONs '
                          'in --output.  --input is not required in this mode.')
    ctl.add_argument('--pytom-dir', default=None,
                     help='Directory containing pytom binaries '
                          '(default: /opt/miniconda3/envs/pytom_tm/bin/)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write aux files and print commands without running pytom')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def _build_extract_cmd(extract_bin, job_json, args):
    """Build pytom_extract_candidates.py command."""
    cmd = [extract_bin, '-j', str(job_json), '-n', str(args.n_particles)]
    if args.particle_diameter:
        cmd += ['--particle-diameter', str(args.particle_diameter)]
    if args.tophat_filter:
        cmd += ['--tophat-filter']
    if args.tophat_bins is not None:
        cmd += ['--tophat-bins', str(args.tophat_bins)]
    if args.cut_off is not None:
        cmd += ['--cut-off', str(args.cut_off)]
    if args.n_false_positives is not None:
        cmd += ['--number-of-false-positives', str(args.n_false_positives)]
    if args.relion5_compat:
        cmd += ['--relion5-compat']
    if getattr(args, 'log', None):
        cmd += ['--log', args.log]
    return cmd


def _print_cmd(cmd):
    it = iter(cmd)
    lines = ['  $ ' + next(it)]
    for tok in it:
        if tok.startswith('-'):
            lines.append('      ' + tok)
        else:
            lines[-1] += '  ' + tok
    print(' \\\n'.join(lines))


def run(args):
    out_dir = Path(args.output).resolve()
    sep     = '─' * 70

    extract_only = getattr(args, 'extract_only', False)

    if extract_only:
        _run_extract_only(args, out_dir, sep)
        return

    # ── Matching mode ─────────────────────────────────────────────────────
    if args.input is None:
        print('ERROR: --input is required unless --extract-only is set')
        sys.exit(1)

    in_dir = Path(args.input).resolve()
    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # Locate pytom binary
    pytom_bin = _find_pytom(args.pytom_dir)
    if not pytom_bin:
        msg = (f'pytom_match_template.py not found.\n'
               f'  Expected at {_PYTOM_BIN}\n'
               f'  Or specify: --pytom-dir /path/to/pytom_tm/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            pytom_bin = 'pytom_match_template.py'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    # ── Find tomogram prefixes ─────────────────────────────────────────────
    if args.vol_suffix:
        vol_glob = f'ts-*{args.vol_suffix}_Vol.mrc'
    else:
        vol_glob = 'ts-*_Vol.mrc'

    vols = [v for v in sorted(in_dir.glob(vol_glob))
            if '_EVN' not in v.name and '_ODD' not in v.name]
    if not vols and not args.vol_suffix:
        vols = [v for v in sorted(in_dir.glob('ts-*.mrc'))
                if not any(tag in v.name for tag in ('_EVN', '_ODD', '_CTF'))]

    if not vols:
        print(f'ERROR: no tomogram volumes found in {in_dir}/ '
              f'(pattern: {vol_glob})')
        sys.exit(1)

    def _prefix_from_vol(vol, vol_suffix):
        name = vol.stem
        for tag in ('_Vol', vol_suffix):
            if tag and name.endswith(tag):
                name = name[:-len(tag)]
        return name

    prefixes = [_prefix_from_vol(v, args.vol_suffix) for v in vols]

    # ── include / exclude filtering ────────────────────────────────────────
    if args.include:
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in inc)]
    if args.exclude:
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in exc)]

    # ── select-ts filter ───────────────────────────────────────────────────
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

    # ── QC setup ──────────────────────────────────────────────────────────
    do_qc    = getattr(args, 'analyse', False)
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    if do_qc:
        try:
            from aretomo3_preprocess.shared.volume_qc import (
                central_slab_projection, projection_to_b64png, make_comparison_html,
            )
        except ImportError as e:
            print(f'WARNING: --analyse requires mrcfile and matplotlib ({e}); skipping report')
            do_qc = False

    # ── Process each tomogram ──────────────────────────────────────────────
    ok, failed = [], []

    for i, prefix in enumerate(prefixes):
        print(f'\n[{i+1}/{len(prefixes)}] {prefix}')

        tomo = _find_tomogram(in_dir, prefix, args.vol_suffix)
        if tomo is None:
            print(f'  WARNING: volume not found for {prefix} — skipping')
            failed.append(prefix)
            continue

        try:
            tlt_out, defocus_out, exposure_out = _read_ts_metadata(
                in_dir, prefix, args.dose
            )
        except (FileNotFoundError, ValueError) as e:
            print(f'  WARNING: {e} — skipping')
            failed.append(prefix)
            continue

        if args.dry_run:
            od = out_dir / prefix
            tlt_file = od / f'{prefix}.tlt'
            exp_file = od / f'{prefix}_exposure.txt'
            df_file  = od / f'{prefix}_defocus.txt' if defocus_out is not None else None
        else:
            tlt_file, df_file, exp_file = _write_aux_files(
                out_dir, prefix, tlt_out, defocus_out, exposure_out
            )

        bmask = None
        if args.bmask_dir:
            candidate = Path(args.bmask_dir) / f'{prefix}.mrc'
            if candidate.exists():
                bmask = candidate

        out_subdir = out_dir / prefix

        cmd = _build_cmd(
            pytom_bin, tomo, tlt_file, df_file, exp_file,
            out_subdir, args, bmask=bmask,
        )

        _print_cmd(cmd)

        # ── Inline extraction command (printed even in dry-run) ───────────
        if args.extract and args.n_particles is not None:
            if args.imod and not args.relion5_compat:
                args.relion5_compat = True
            extract_bin = _find_pytom_extract(args.pytom_dir)
            if not extract_bin:
                extract_bin = 'pytom_extract_candidates.py'
            job_json_dry = out_subdir / f'{prefix}_job.json'
            ecmd_dry = _build_extract_cmd(extract_bin, job_json_dry, args)
            print('\n  Extraction command:')
            _print_cmd(ecmd_dry)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(prefix)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: pytom exited with code {ret.returncode}')
            failed.append(prefix)
            continue

        ok.append(prefix)

        # ── QC (score map) ────────────────────────────────────────────────
        if do_qc:
            before_b64 = after_b64 = None
            proj = central_slab_projection(tomo, qc_thick)
            if proj:
                before_b64 = projection_to_b64png(proj['img'])
            score_files = sorted(out_subdir.glob('*_scores.mrc'))
            if score_files:
                sproj = central_slab_projection(score_files[0], qc_thick)
                if sproj:
                    after_b64 = projection_to_b64png(sproj['img'], pct=(50, 99.9))
            qc_entries.append({
                'ts_name':     prefix,
                'before_b64':  before_b64,
                'after_b64':   after_b64,
                'before_path': str(tomo),
                'after_path':  str(score_files[0]) if score_files else '',
                'metadata': {
                    'template': Path(args.template).name,
                    'voxel':    f'{args.voxel_size} Å',
                },
            })

        # ── Inline extraction ─────────────────────────────────────────────
        if args.extract:
            if args.n_particles is None:
                print('  WARNING: --extract set but --n-particles not given — skipping')
            else:
                job_jsons = sorted(out_subdir.glob('*_job.json'))
                if not job_jsons:
                    print(f'  WARNING: no *_job.json in {out_subdir} — skipping extraction')
                else:
                    extract_bin = _find_pytom_extract(args.pytom_dir)
                    if not extract_bin:
                        print('  WARNING: pytom_extract_candidates.py not found')
                    else:
                        job_json = job_jsons[0]
                        ecmd = _build_extract_cmd(extract_bin, job_json, args)
                        print('\n  Extracting candidates...')
                        eret = subprocess.run(ecmd)
                        if eret.returncode != 0:
                            print(f'  WARNING: extraction exited with code {eret.returncode}')
                        elif args.imod:
                            mod_dir = (Path(args.imod_dir).resolve()
                                       if args.imod_dir else out_dir / 'mod')
                            for star in sorted(out_subdir.glob('*_particles.star')):
                                _star_to_mod(star, job_json, mod_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')

    # ── QC report ─────────────────────────────────────────────────────────
    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'pytom_match_qc.html')
        make_comparison_html(
            entries      = qc_entries,
            out_path     = html_path,
            title        = 'pytom-match QC',
            command      = ' '.join(sys.argv),
            before_label = 'Tomogram',
            after_label  = 'Score map',
            slab_angst   = qc_thick,
        )

    if args.dry_run:
        return

    update_section(
        section='pytom_match',
        values={
            'command':     ' '.join(sys.argv),
            'args':        args_to_dict(args),
            'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
            'n_processed': len(ok),
            'failed':      failed,
            'output_dir':  str(out_dir),
        },
        backup_dir=out_dir,
    )


def _run_extract_only(args, out_dir, sep):
    """Run extraction on existing job JSONs in out_dir."""
    if not out_dir.is_dir():
        print(f'ERROR: --output {out_dir} not found (run template matching first)')
        sys.exit(1)

    if args.n_particles is None:
        print('ERROR: --n-particles is required with --extract-only')
        sys.exit(1)

    if args.imod and not args.relion5_compat:
        print('WARNING: --imod requires --relion5-compat; adding automatically.')
        args.relion5_compat = True

    extract_bin = _find_pytom_extract(args.pytom_dir)
    if not extract_bin:
        msg = (f'pytom_extract_candidates.py not found.\n'
               f'  Expected at {_PYTOM_EXTRACT_BIN}\n'
               f'  Or specify: --pytom-dir /path/to/pytom_tm/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            extract_bin = 'pytom_extract_candidates.py'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    mod_dir = Path(args.imod_dir).resolve() if args.imod_dir else out_dir / 'mod'

    # Find job JSONs
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    jobs = _find_job_jsons(out_dir, selected_ts)

    if getattr(args, 'include', None):
        inc = args.include[0].split(',') if len(args.include) == 1 else args.include
        jobs = [(n, j) for n, j in jobs
                if any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in inc)]
    if getattr(args, 'exclude', None):
        exc = args.exclude[0].split(',') if len(args.exclude) == 1 else args.exclude
        jobs = [(n, j) for n, j in jobs
                if not any(re.match(f'^{pat.replace("*", ".*")}$', n) for pat in exc)]

    if not jobs:
        print(f'ERROR: no *_job.json files found in {out_dir}/')
        sys.exit(1)

    print(f'Tomograms to extract: {len(jobs)}')
    print(sep)
    for ts_name, _ in jobs[:10]:
        print(f'  {ts_name}')
    if len(jobs) > 10:
        print(f'  ... ({len(jobs) - 10} more)')
    print(sep)

    # ── QC setup ──────────────────────────────────────────────────────────
    do_qc    = getattr(args, 'analyse', False)
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    if do_qc:
        try:
            import warnings as _warnings
            import starfile as _starfile
            from aretomo3_preprocess.shared.volume_qc import (
                slab_with_picks_b64, central_slab_projection,
                projection_to_b64png, make_picks_html,
            )
        except ImportError as e:
            print(f'WARNING: --analyse requires starfile and matplotlib ({e}); skipping')
            do_qc = False

    ok, failed = [], []

    for i, (ts_name, job_json) in enumerate(jobs):
        print(f'\n[{i+1}/{len(jobs)}] {ts_name}')

        cmd = _build_extract_cmd(extract_bin, job_json, args)
        _print_cmd(cmd)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(ts_name)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: extraction exited with code {ret.returncode}')
            failed.append(ts_name)
            continue

        ok.append(ts_name)

        if args.imod:
            star_files = sorted(job_json.parent.glob('*_particles.star'))
            for star in star_files:
                _star_to_mod(star, job_json, mod_dir)

        # ── QC (picks overlay) ────────────────────────────────────────────
        if do_qc:
            with open(job_json) as fh:
                job = json.load(fh)
            tomo_path = job.get('tomogram') or job.get('tomogram_path') or job.get('volume_path')
            star_files  = sorted(job_json.parent.glob('*_particles.star'))
            score_files = sorted(job_json.parent.glob('*_scores.mrc'))
            entry = {
                'ts_name':   ts_name,
                'img_b64':   None,
                'score_b64': None,
                'n_total':   0,
                'n_shown':   0,
                'tomo_path': tomo_path or '',
                'metadata':  {},
            }
            if star_files and tomo_path and Path(tomo_path).exists():
                try:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter('ignore')
                        data = _starfile.read(str(star_files[0]))
                    df = data[next(iter(data))] if isinstance(data, dict) else data
                    result = slab_with_picks_b64(
                        tomo_path, df,
                        slab_angst=qc_thick,
                        particle_diameter=args.particle_diameter,
                    )
                    score_b64 = None
                    if score_files:
                        sproj = central_slab_projection(score_files[0], qc_thick)
                        if sproj:
                            score_b64 = projection_to_b64png(sproj['img'], pct=(50, 99.9))
                    if result:
                        entry.update({
                            'img_b64':   result['img_b64'],
                            'score_b64': score_b64,
                            'n_total':   result['n_total'],
                            'n_shown':   result['n_shown'],
                            'metadata': {
                                'slab':   f'{result["slab_a"]:.0f} Å  ({result["vox"]:.2f} Å/px)',
                                'picked': f'{result["n_shown"]} / {result["n_total"]} in slab',
                            },
                        })
                except Exception as exc:
                    print(f'  WARNING: QC render failed: {exc}')
            qc_entries.append(entry)

    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')
    if args.imod and not args.dry_run and ok:
        print(f'IMOD models: {mod_dir}/')

    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'pytom_extract_qc.html')
        make_picks_html(
            entries    = qc_entries,
            out_path   = html_path,
            title      = 'pytom-extract QC',
            command    = ' '.join(sys.argv),
            slab_angst = qc_thick,
        )

    if args.dry_run:
        return

    update_section(
        section='pytom_extract',
        values={
            'command':     ' '.join(sys.argv),
            'args':        args_to_dict(args),
            'timestamp':   datetime.datetime.now().isoformat(timespec='seconds'),
            'n_processed': len(ok),
            'failed':      failed,
            'output_dir':  str(out_dir),
        },
        backup_dir=out_dir,
    )
