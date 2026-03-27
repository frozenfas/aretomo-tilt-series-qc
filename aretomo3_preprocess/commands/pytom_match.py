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

After template matching, extract candidates with:
  pytom_extract_candidates.py \\
      --tomogram-star pytom_match/pytom_match.star \\
      --number-of-particles 2000 \\
      --radius 150
"""

import re
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
    import shutil
    candidates = []
    if pytom_dir:
        candidates.append(str(Path(pytom_dir) / 'pytom_match_template.py'))
    candidates.append(_PYTOM_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('pytom_match_template.py')


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
    inp.add_argument('--input', '-i', required=True,
                     help='AreTomo3 output directory')
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

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--output', '-o', default='pytom_match',
                     help='Output directory (per-TS subdirectories are created inside)')
    ctl.add_argument('--pytom-dir', default=None,
                     help='Directory containing pytom_match_template.py '
                          '(default: /opt/miniconda3/envs/pytom_tm/bin/)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write aux files and print commands without running pytom')

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
        # Fallback: ts-*.mrc (older AreTomo3 output)
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

        # Print command with one flag+value pair per line for readability
        it = iter(cmd)
        lines = ['  $ ' + next(it)]
        for tok in it:
            if tok.startswith('-'):
                lines.append('      ' + tok)
            else:
                lines[-1] += '  ' + tok
        print(' \\\n'.join(lines))

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(prefix)
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: pytom exited with code {ret.returncode}')
            failed.append(prefix)
        else:
            ok.append(prefix)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print(f'Done.  {len(ok)} succeeded, {len(failed)} failed.')
    if failed:
        print(f'Failed: {", ".join(failed)}')

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
