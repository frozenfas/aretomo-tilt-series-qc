"""
run-aretomo3-per-ts — run AreTomo3 individually per tilt series using
per-lamella median TiltAxis and AlignZ from a prior analyse run.

After running analyse (which groups tilt series by lamella and computes
per-lamella statistics), use this command to re-run AreTomo3 with
--Cmd 1 (alignment + reconstruction, no motion correction) supplying the
lamella-specific tilt axis and slab thickness for each tilt series.

Reads per-lamella stats from aretomo3_project.json (written by analyse)
and lamella_positions.csv to look up TiltAxis/AlignZ per TS.

AreTomo3 is invoked once per tilt series; output is streamed live and each
TS gets its own log file inside the output directory.

Typical usage
-------------
    aretomo3-preprocess run-aretomo3-per-ts \\
        --mrcdir /path/to/relion/stacks \\
        --output run002 \\
        --analysis run001_analysis \\
        --mdocdir frames \\
        --gpu 2 3 \\
        --apix 1.63 \\
        --dry-run
"""

import csv
import json
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.commands.run_aretomo3 import _fmt_command, _num
from aretomo3_preprocess.shared.project_json import update_section, args_to_dict


# ─────────────────────────────────────────────────────────────────────────────
# Lamella parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lamella_params(analysis_dir: Path) -> tuple[dict, dict]:
    """
    Load per-lamella rot/AlignZ and the ts→lamella mapping.

    Returns (lamella_params, ts_to_lamella) where:
      lamella_params  — {lamella_id_str: {'rot_deg': float, 'align_z_px': int}}
      ts_to_lamella   — {ts_name: lamella_id_str}

    Prefers the pre-computed lamella_suggested from aretomo3_project.json.
    Falls back to computing medians directly from alignment_data.json.
    """
    import numpy as np

    csv_path  = analysis_dir / 'lamella_positions.csv'
    json_path = analysis_dir / 'alignment_data.json'
    proj_path = analysis_dir / 'aretomo3_project.json'

    if not csv_path.exists():
        raise FileNotFoundError(f'lamella_positions.csv not found in {analysis_dir}')

    # ts_name → lamella id (stored as string to match project JSON keys)
    ts_to_lamella = {}
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            ts_to_lamella[row['ts_name']] = str(int(row['lamella']))

    # Try pre-computed stats first
    if proj_path.exists():
        with open(proj_path) as fh:
            proj = json.load(fh)
        lamella_stats = proj.get('analyse', {}).get('lamella_suggested', {})
        if lamella_stats:
            # Ensure we have the keys we need
            lamella_params = {
                lam: {
                    'rot_deg':   stats['rot_deg'],
                    'align_z_px': stats['align_z_px'],
                }
                for lam, stats in lamella_stats.items()
                if 'rot_deg' in stats and 'align_z_px' in stats
            }
            if lamella_params:
                return lamella_params, ts_to_lamella

    # Fallback: compute from alignment_data.json
    if not json_path.exists():
        raise FileNotFoundError(
            f'Neither aretomo3_project.json nor alignment_data.json found in {analysis_dir}'
        )
    with open(json_path) as fh:
        all_ts = json.load(fh)

    rots_by_lam   = {}
    z_by_lam      = {}
    for ts_name, data in all_ts.items():
        lam = ts_to_lamella.get(ts_name)
        if lam is None:
            continue
        if data.get('frames'):
            rot = data['frames'][0].get('rot')
            if rot is not None:
                rots_by_lam.setdefault(lam, []).append(rot)
        z = data.get('thickness')
        if z is not None:
            z_by_lam.setdefault(lam, []).append(z)

    lamella_params = {}
    for lam in set(list(rots_by_lam) + list(z_by_lam)):
        lamella_params[lam] = {
            'rot_deg':    round(float(np.median(rots_by_lam[lam])), 2)
                          if lam in rots_by_lam else None,
            'align_z_px': int(round(np.median(z_by_lam[lam])))
                          if lam in z_by_lam else None,
        }
    return lamella_params, ts_to_lamella


# ─────────────────────────────────────────────────────────────────────────────
# Per-TS command builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_per_ts_cmd(args, mrc_path: Path, out_mrc: Path, mdoc_path: Path,
                      rot: float, align_z: int) -> list:
    """Build the AreTomo3 command for a single tilt series."""
    vol_z = args.vol_z if args.vol_z is not None else align_z

    cmd = [args.aretomo3_bin]
    cmd += ['-InMrc',  str(mrc_path)]
    cmd += ['-OutMrc', str(out_mrc)]
    if mdoc_path.exists():
        cmd += ['-Mdoc', str(mdoc_path)]
    cmd += ['-Cmd',      _num(args.cmd)]
    cmd += ['-TiltAxis', _num(rot), _num(args.tilt_axis_search)]
    cmd += ['-AlignZ',   _num(align_z)]
    cmd += ['-VolZ',     _num(vol_z)]
    cmd += ['-Gpu']    + [_num(g) for g in args.gpu]
    cmd += ['-PixSize',     _num(args.apix)]
    cmd += ['-Kv',          _num(args.kv)]
    cmd += ['-Cs',          _num(args.cs)]
    cmd += ['-AmpContrast', _num(args.amp_contrast)]
    cmd += ['-AtBin']  + [_num(v) for v in args.at_bin]
    cmd += ['-AtPatch'] + [_num(v) for v in args.at_patch]
    cmd += ['-Wbp',      _num(args.wbp)]
    cmd += ['-FlipVol',  _num(args.flip_vol)]
    cmd += ['-TiltCor',  _num(args.tilt_cor)]
    cmd += ['-DarkTol',  _num(args.dark_tol)]
    cmd += ['-CorrCTF',  _num(args.corr_ctf)]
    cmd += ['-OutXF',    _num(args.out_xf)]
    cmd += ['-OutImod',  _num(args.out_imod)]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'run-aretomo3-per-ts',
        help='Run AreTomo3 per-TS using per-lamella TiltAxis and AlignZ from analyse',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--mrcdir',   '-m', required=True,
                     help='Directory containing ts-xxx.mrc input stacks')
    req.add_argument('--output',   '-o', required=True,
                     help='Output directory for AreTomo3 results')
    req.add_argument('--analysis', '-A', required=True,
                     help='analyse output directory (lamella_positions.csv + '
                          'aretomo3_project.json)')
    req.add_argument('--gpu',      '-G', type=int, nargs='+', required=True,
                     metavar='ID',
                     help='GPU ID(s) to use (-Gpu)')
    req.add_argument('--apix',     '-a', type=float, required=True,
                     help='Pixel size in Å/px (-PixSize)')

    inp = p.add_argument_group('input')
    inp.add_argument('--mdocdir', default='frames',
                     help='Directory containing ts-xxx.mdoc files (-Mdoc)')
    inp.add_argument('--in-skips', nargs='+',
                     default=['_Vol', '_CTF', '_EVN', '_ODD'],
                     metavar='PATTERN',
                     help='Stem substrings to exclude from --mrcdir; '
                          'default excludes AreTomo3 side files '
                          '(_Vol _CTF _EVN _ODD). Pass an empty string to '
                          'disable, or add extra patterns e.g. ts-005')

    mic = p.add_argument_group('microscope / acquisition')
    mic.add_argument('--kv',           type=float, default=300.0,
                     help='Accelerating voltage in kV (-Kv)')
    mic.add_argument('--cs',           type=float, default=2.7,
                     help='Spherical aberration in mm (-Cs)')
    mic.add_argument('--amp-contrast', type=float, default=0.1,
                     help='Amplitude contrast ratio (-AmpContrast)')
    mic.add_argument('--cmd',          type=int,   default=1,
                     help='AreTomo3 pipeline mode; 1=alignment+recon (-Cmd)')

    ali = p.add_argument_group('alignment and reconstruction')
    ali.add_argument('--tilt-axis-search', type=float, default=3.0,
                     help='TiltAxis search range in degrees passed to AreTomo3 '
                          '(-TiltAxis <rot> <search>); 0 = fix axis exactly')
    ali.add_argument('--vol-z',    type=int,   default=None,
                     help='Reconstruction Z in px; defaults to per-lamella AlignZ (-VolZ)')
    ali.add_argument('--at-bin',   type=float, nargs='+', default=[4.0],
                     metavar='BIN',
                     help='Tomogram binning; up to 3 values (-AtBin)')
    ali.add_argument('--at-patch', type=int,   nargs=2,   default=[0, 0],
                     metavar=('X', 'Y'),
                     help='Local alignment patch grid; 0 0=global only (-AtPatch)')
    ali.add_argument('--wbp',      type=int,   default=1,
                     help='Reconstruction: 1=WBP 0=SART (-Wbp)')
    ali.add_argument('--flip-vol', type=int,   default=1,
                     help='Flip reconstructed volume (-FlipVol)')
    ali.add_argument('--tilt-cor', type=int,   default=1,
                     help='Apply tilt angle offset correction (-TiltCor)')
    ali.add_argument('--dark-tol', type=float, default=0.7,
                     help='Dark frame rejection tolerance (-DarkTol)')
    ali.add_argument('--corr-ctf', type=int,   default=1,
                     help='CTF correction before reconstruction (-CorrCTF)')
    ali.add_argument('--out-xf',   type=int,   default=1,
                     help='Write IMOD XF transform files (-OutXF)')
    ali.add_argument('--out-imod', type=int,   default=1,
                     help='Write IMOD support files for RELION (-OutImod)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--aretomo3', dest='aretomo3_bin', default='AreTomo3',
                     help='Path to or name of the AreTomo3 executable')
    ctl.add_argument('--overwrite', action='store_true',
                     help='Re-run even if output .aln already exists')
    ctl.add_argument('--dry-run',   action='store_true',
                     help='Print per-TS commands without executing')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    mrc_dir  = Path(args.mrcdir)
    out_dir  = Path(args.output)
    ana_dir  = Path(args.analysis)
    mdoc_dir = Path(args.mdocdir)

    for d, name in [(mrc_dir, '--mrcdir'), (ana_dir, '--analysis')]:
        if not d.is_dir():
            print(f'Error: {name} {d} not found')
            sys.exit(1)

    # ── Load per-lamella parameters ────────────────────────────────────────
    try:
        lamella_params, ts_to_lamella = _load_lamella_params(ana_dir)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)

    print(f'Per-lamella parameters (from {ana_dir}):')
    for lam_id in sorted(lamella_params, key=lambda x: int(x)):
        p = lamella_params[lam_id]
        vol_z = args.vol_z if args.vol_z is not None else p['align_z_px']
        print(f'  Lamella {lam_id:>2}  '
              f'TiltAxis={p["rot_deg"]}°  '
              f'AlignZ={p["align_z_px"]} px  '
              f'VolZ={vol_z} px')
    print()

    # ── Find input stacks ──────────────────────────────────────────────────
    all_mrc = sorted(mrc_dir.glob('ts-*.mrc'))
    if not all_mrc:
        print(f'No ts-*.mrc files found in {mrc_dir}')
        sys.exit(1)

    # Apply --in-skips filter (default excludes AreTomo3 side files)
    skip_pats = [p for p in (args.in_skips or []) if p]
    if skip_pats:
        mrc_files = [p for p in all_mrc
                     if not any(pat in p.stem for pat in skip_pats)]
        n_excluded = len(all_mrc) - len(mrc_files)
        print(f'--in-skips {skip_pats}: {n_excluded} files excluded '
              f'({len(mrc_files)} remaining)\n')
    else:
        mrc_files = all_mrc

    if not args.dry_run:
        if shutil.which(args.aretomo3_bin) is None \
                and not Path(args.aretomo3_bin).is_file():
            print(f'Error: AreTomo3 binary not found: {args.aretomo3_bin!r}')
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    mode = 'DRY RUN' if args.dry_run else 'RUN'
    print(f'Found {len(mrc_files)} stacks in {mrc_dir}  '
          f'({mode}, output → {out_dir})\n')

    sep = '─' * 70
    n_run = n_skip = n_fail = 0
    run_log = []   # (ts_name, lamella, rot, align_z, returncode)

    for mrc_path in mrc_files:
        ts_name  = mrc_path.stem
        out_mrc  = out_dir  / f'{ts_name}.mrc'
        out_aln  = out_dir  / f'{ts_name}.aln'
        mdoc_path = mdoc_dir / f'{ts_name}.mdoc'

        # Skip if already done
        if out_aln.exists() and not args.overwrite:
            print(f'  SKIP  {ts_name}  (.aln exists — use --overwrite to re-run)')
            n_skip += 1
            continue

        # Lamella lookup
        lam_id = ts_to_lamella.get(ts_name)
        if lam_id is None:
            print(f'  WARN  {ts_name}  not in lamella_positions.csv — skipping')
            n_skip += 1
            continue

        lp = lamella_params.get(lam_id, {})
        rot     = lp.get('rot_deg')
        align_z = lp.get('align_z_px')
        if rot is None or align_z is None:
            print(f'  WARN  {ts_name}  lamella {lam_id} missing rot/AlignZ — skipping')
            n_skip += 1
            continue

        cmd = _build_per_ts_cmd(args, mrc_path, out_mrc, mdoc_path, rot, align_z)

        print(sep)
        print(f'  {ts_name}  [lamella {lam_id}  TiltAxis={rot}°  AlignZ={align_z} px]')
        print()
        print(_fmt_command(cmd, annotate=False))
        print()

        if args.dry_run:
            n_run += 1
            continue

        # ── Run and stream output ──────────────────────────────────────────
        ts_log_path = out_dir / f'{ts_name}_aretomo3.log'
        with open(ts_log_path, 'w') as log_fh:
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

        rc = proc.returncode
        run_log.append((ts_name, lam_id, rot, align_z, rc))

        if rc != 0:
            print(f'\n  FAILED  {ts_name}  (exit code {rc}  —  log: {ts_log_path})')
            n_fail += 1
        else:
            print(f'\n  OK  {ts_name}')
            n_run += 1

    print(sep)
    print(f'\nDone: {n_run} {"would run" if args.dry_run else "run"}, '
          f'{n_skip} skipped, {n_fail} failed\n')

    if not args.dry_run:
        update_section(
            section='run_aretomo3_per_ts',
            values={
                'command':    ' '.join(sys.argv),
                'args':       args_to_dict(args),
                'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
                'output_dir': str(out_dir.resolve()),
                'n_run':      n_run,
                'n_skip':     n_skip,
                'n_fail':     n_fail,
                'ts_results': [
                    {'ts': ts, 'lamella': lam, 'rot': rot,
                     'align_z': z, 'returncode': rc}
                    for ts, lam, rot, z, rc in run_log
                ],
            },
        )
