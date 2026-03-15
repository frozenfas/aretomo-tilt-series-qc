"""
ddw-train   — prepare training subtomograms and fit a U-Net model for
              missing-wedge reconstruction and denoising (DeepDeWedge steps 1–2).
ddw-refine  — apply a fitted model to denoise full tomograms (step 3).

DeepDeWedge (DDW) trains a self-supervised U-Net on even/odd half-set volume
pairs using a noise2noise approach.  EVN/ODD volumes must be produced by
AreTomo3 with -SplitSum 1 (--split-sum 1).

Workflow
--------
  1. run-aretomo3 cmd=2 --split-sum 1  →  ts-xxx_EVN_Vol.mrc + ts-xxx_ODD_Vol.mrc
  2. aretomo3-preprocess ddw-train     →  trains one U-Net on the full dataset
  3. aretomo3-preprocess ddw-refine    →  applies model to produce denoised volumes

One model is trained on ALL provided volumes (or a stratified subset via
--n-vols); the same model is then applied to every tomogram in ddw-refine.

EVN/ODD pair detection
----------------------
  Single --at-bin run:   ts-xxx_EVN_Vol.mrc / ts-xxx_ODD_Vol.mrc  (default)
  Multiple --at-bin run: ts-xxx_b8_EVN.mrc  / ts-xxx_b8_ODD.mrc   (--vol-suffix _b8)

Missing wedge angle
-------------------
  Auto-detected from _TLT.txt if not given:  mw_angle = 90 - max(|tilt|)
  For a ±60° tilt series: mw_angle = 30.  For ±70°: mw_angle = 20.

Typical usage
-------------
  # Step 1+2: extract subtomograms and train (takes several hours)
  aretomo3-preprocess ddw-train \\
      --input run005-cmd2-sart-thr80/ \\
      --output run005-ddw/ \\
      --gpu 0 1 \\
      --dry-run

  # Step 3: apply model to all TS
  aretomo3-preprocess ddw-refine \\
      --input run005-cmd2-sart-thr80/ \\
      --project-dir run005-ddw/ \\
      --gpu 0 \\
      --dry-run
"""

import re
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

import yaml

from aretomo3_preprocess.shared.project_json import (
    load_or_create, update_section, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.commands.cryocare import (
    _find_evn_odd_pairs,
    _load_tsselect_defocus,
    _stratified_sample,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mw_angle(in_dir: Path):
    """
    Estimate missing wedge angle from .aln files in in_dir.

    Reads the TILT column (col 9) from data rows of each .aln — these are
    the corrected tilt angles of frames that were actually aligned and used
    in the reconstruction (dark frames are excluded from .aln data rows).
    This gives the true angular range of the reconstruction, not the nominal
    acquisition range.

    MW angle = 90 - max(|tilt|), rounded to nearest integer.
    Samples the first 10 .aln files to estimate the range.
    Returns None if no .aln files found.
    """
    aln_files = sorted(in_dir.glob('ts-*.aln'))[:10]
    if not aln_files:
        return None

    max_tilt = 0.0
    n_read   = 0
    for aln_file in aln_files:
        try:
            with open(aln_file) as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith('#') or not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) == 10:
                        try:
                            max_tilt = max(max_tilt, abs(float(parts[9])))
                        except ValueError:
                            pass
            n_read += 1
        except Exception:
            pass

    if n_read == 0:
        return None
    return max(0, int(round(90.0 - max_tilt)))


def _find_best_checkpoint(logdir: Path):
    """
    Find the best model checkpoint in logdir.

    Prefers checkpoints containing 'val_loss' in the name (lowest wins).
    Falls back to most recently modified .ckpt file.
    Returns None if no checkpoints found.
    """
    ckpts = list(logdir.rglob('*.ckpt'))
    if not ckpts:
        return None

    val_loss_ckpts = []
    for ckpt in ckpts:
        m = re.search(r'val.loss[=_]([\d.eE+\-]+)', ckpt.name)
        if m:
            try:
                val_loss_ckpts.append((float(m.group(1)), ckpt))
            except ValueError:
                pass

    if val_loss_ckpts:
        return sorted(val_loss_ckpts, key=lambda x: x[0])[0][1]

    return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, 'w') as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


def _run_ddw(ddw_bin: str, subcmd: str, config_path: Path,
             extra_args=None, dry_run: bool = False) -> None:
    """Run `ddw <subcmd> --config <config_path> [extra_args]`, streaming stdout."""
    cmd = [ddw_bin, subcmd, '--config', str(config_path)]
    if extra_args:
        cmd += extra_args
    print(f'  $ {" ".join(cmd)}\n')
    if dry_run:
        return
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f'ERROR: ddw {subcmd} exited with code {ret.returncode}')
        sys.exit(ret.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# ddw-train  (prepare-data + fit-model)
# ─────────────────────────────────────────────────────────────────────────────

def _add_train_parser(subparsers):
    p = subparsers.add_parser(
        'ddw-train',
        help='Extract subtomograms and train DeepDeWedge U-Net (steps 1–2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_EVN_Vol.mrc / _ODD_Vol.mrc')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume suffix; empty=auto-detect (_EVN_Vol first, '
                          'then _EVN). Use e.g. "_b8" for multi-bin runs.')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; filters volumes and provides '
                          'ref_defocus_um for stratified training subset selection')
    inp.add_argument('--n-vols', type=int, default=None, metavar='N',
                     help='Volumes to use for training, stratified by defocus. '
                          'Default: use all volumes.')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', required=True,
                     help='Project directory (subtomos/, logs/ created here)')

    ddw = p.add_argument_group('DeepDeWedge parameters')
    ddw.add_argument('--mw-angle', type=int, default=None, metavar='DEG',
                     help='Missing wedge width in degrees. '
                          'Auto-detected from .aln files if omitted '
                          '(= 90 - max aligned tilt, excluding dark frames).')
    ddw.add_argument('--subtomo-size', type=int, default=96,
                     help='Cubic subtomogram size in voxels. '
                          'Must be divisible by 2^unet-depth (default 8).')
    ddw.add_argument('--num-epochs', type=int, default=1000,
                     help='Training epochs')
    ddw.add_argument('--batch-size', type=int, default=5,
                     help='Training batch size')
    ddw.add_argument('--lr', type=float, default=0.0004,
                     help='Adam learning rate')
    ddw.add_argument('--unet-chans', type=int, default=64,
                     help='Feature channels in first U-Net layer')
    ddw.add_argument('--unet-depth', type=int, default=3,
                     help='U-Net downsampling depth '
                          '(subtomo-size must be divisible by 2^depth)')
    ddw.add_argument('--val-fraction', type=float, default=0.1,
                     help='Fraction of subtomograms held out for validation')
    ddw.add_argument('--num-workers', type=int, default=8,
                     help='CPU workers for data loading')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, nargs='+', default=[0], metavar='ID',
                     help='GPU ID(s); multiple IDs use multi-GPU training')
    ctl.add_argument('--seed', type=int, default=42,
                     help='Random seed for reproducibility')
    ctl.add_argument('--ddw', dest='ddw_bin',
                     default='/opt/miniconda3/envs/ddw_env/bin/ddw',
                     help='Path to ddw executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write config without running ddw')

    p.set_defaults(func=_run_train)
    return p


def _run_train(args):
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    sep     = '─' * 70
    prefix  = '[DRY RUN] ' if args.dry_run else ''

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # ── Find EVN/ODD pairs ─────────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no EVN/ODD volume pairs found in {in_dir}/')
        print('       Run-aretomo3 cmd=2 with --split-sum 1 first.')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Apply TS selection filter ──────────────────────────────────────────
    csv_path    = getattr(args, 'select_ts', None)
    selected_ts = resolve_selected_ts(csv_path)
    if selected_ts is not None:
        orig_n = len(pairs)
        pairs  = [p for p in pairs if p['ts_name'] in selected_ts]
        n_excl = orig_n - len(pairs)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(pairs)} remaining')
        if not pairs:
            print('ERROR: no volumes remain after TS selection filter')
            sys.exit(1)

    # ── Optionally subset for training ────────────────────────────────────
    if args.n_vols is not None and args.n_vols < len(pairs):
        defocus    = _load_tsselect_defocus(csv_path)
        candidates = [
            {'ts_name': p['ts_name'], 'evn': p['evn'], 'odd': p['odd'],
             'defocus_um': defocus.get(p['ts_name'])}
            for p in pairs
        ]
        selected_names = _stratified_sample(candidates, args.n_vols, seed=args.seed)
        train_pairs = [p for p in pairs if p['ts_name'] in set(selected_names)]
        print(f'Training subset: {len(train_pairs)} / {len(pairs)} volumes '
              f'(stratified by defocus, --n-vols {args.n_vols})')
    else:
        train_pairs = pairs
        print(f'Training on all {len(train_pairs)} volumes')

    print(sep)
    for p in train_pairs[:10]:
        print(f'  {p["ts_name"]}  {p["evn"].name}')
    if len(train_pairs) > 10:
        print(f'  ... ({len(train_pairs) - 10} more)')
    print(sep)

    # ── Auto-detect missing wedge angle ───────────────────────────────────
    mw_angle = args.mw_angle
    if mw_angle is None:
        mw_angle = _detect_mw_angle(in_dir)
        if mw_angle is not None:
            print(f'Missing wedge angle : {mw_angle}° (auto-detected from .aln)')
        else:
            print('ERROR: could not auto-detect missing wedge angle.')
            print('       Pass --mw-angle explicitly (= 90 - max_tilt_angle).')
            sys.exit(1)
    else:
        print(f'Missing wedge angle : {mw_angle}° (explicit --mw-angle)')

    # ── Validate subtomo-size / unet-depth compatibility ─────────────────
    div = 2 ** args.unet_depth
    if args.subtomo_size % div != 0:
        print(f'ERROR: --subtomo-size {args.subtomo_size} is not divisible by '
              f'2^{args.unet_depth} = {div}')
        sys.exit(1)

    # ── Build DDW config YAML ──────────────────────────────────────────────
    gpu_val = args.gpu[0] if len(args.gpu) == 1 else args.gpu

    config = {
        'shared': {
            'project_dir':  str(out_dir.resolve()),
            'tomo0_files':  [str(p['evn'].resolve()) for p in train_pairs],
            'tomo1_files':  [str(p['odd'].resolve()) for p in train_pairs],
            'subtomo_size': args.subtomo_size,
            'mw_angle':     mw_angle,
            'num_workers':  args.num_workers,
            'gpu':          gpu_val,
            'seed':         args.seed,
        },
        'prepare_data': {
            'val_fraction': args.val_fraction,
        },
        'fit_model': {
            'unet_params_dict': {
                'chans':                args.unet_chans,
                'num_downsample_layers': args.unet_depth,
                'drop_prob':            0.0,
            },
            'adam_params_dict': {
                'lr': args.lr,
            },
            'num_epochs':  args.num_epochs,
            'batch_size':  args.batch_size,
            'update_subtomo_missing_wedges_every_n_epochs': 10,
            'check_val_every_n_epochs':                     10,
            'save_n_models_with_lowest_val_loss':           5,
            'save_n_models_with_lowest_fitting_loss':       5,
            'save_model_every_n_epochs':                    50,
            'logger': 'csv',
        },
    }

    config_path = out_dir / 'ddw_config.yaml'
    print(f'\n{prefix}Config  : {config_path}')
    print(f'{prefix}Subtomo : {out_dir}/subtomos/')
    print(f'{prefix}Logs    : {out_dir}/logs/')

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(config_path, config)

        if shutil.which(args.ddw_bin) is None and not Path(args.ddw_bin).is_file():
            print(f'ERROR: ddw binary not found: {args.ddw_bin!r}')
            print('       Install DeepDeWedge or use --ddw /path/to/ddw')
            sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────
    print(f'\n{prefix}Step 1: prepare-data')
    _run_ddw(args.ddw_bin, 'prepare-data', config_path, dry_run=args.dry_run)

    print(f'\n{prefix}Step 2: fit-model')
    _run_ddw(args.ddw_bin, 'fit-model', config_path, dry_run=args.dry_run)

    if args.dry_run:
        return

    print(f'\nDeepDeWedge training complete.')
    print(f'  Config      : {config_path}')
    print(f'  Checkpoints : {out_dir}/logs/')
    print(f'\n  Next step:')
    print(f'    aretomo3-preprocess ddw-refine \\')
    print(f'        --input {in_dir} --project-dir {out_dir} --gpu 0')

    update_section(
        section='ddw_train',
        values={
            'command':    ' '.join(sys.argv),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'n_vols':     len(train_pairs),
            'selected':   [p['ts_name'] for p in train_pairs],
            'output_dir': str(out_dir.resolve()),
            'config':     str(config_path.resolve()),
            'mw_angle':   mw_angle,
        },
        backup_dir=out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ddw-refine  (refine-tomogram)
# ─────────────────────────────────────────────────────────────────────────────

def _add_refine_parser(subparsers):
    p = subparsers.add_parser(
        'ddw-refine',
        help='Denoise tomograms with a fitted DeepDeWedge model (step 3)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_EVN_Vol.mrc / _ODD_Vol.mrc')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume suffix; empty=auto-detect')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only selected TS are refined')
    inp.add_argument('--project-dir', default=None,
                     help='ddw-train output directory (contains logs/ and '
                          'ddw_config.yaml). Auto-read from project.json if omitted.')
    inp.add_argument('--checkpoint', default=None, metavar='CKPT',
                     help='Model checkpoint (.ckpt). '
                          'Auto-detected from <project-dir>/logs/ if not given '
                          '(lowest val_loss wins).')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default=None,
                     help='Output directory for refined volumes. '
                          'Default: <project-dir>/refined/')

    ddw = p.add_argument_group('DeepDeWedge parameters')
    ddw.add_argument('--mw-angle', type=int, default=None, metavar='DEG',
                     help='Missing wedge angle. '
                          'Auto-read from ddw_config.yaml if not given.')
    ddw.add_argument('--subtomo-size', type=int, default=None,
                     help='Subtomogram size used in training. '
                          'Auto-read from ddw_config.yaml if not given.')
    ddw.add_argument('--batch-size', type=int, default=10,
                     help='Batch size for inference')
    ddw.add_argument('--num-workers', type=int, default=4,
                     help='CPU workers for data loading')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, default=0, metavar='ID',
                     help='GPU ID (single GPU only for refinement)')
    ctl.add_argument('--ddw', dest='ddw_bin',
                     default='/opt/miniconda3/envs/ddw_env/bin/ddw',
                     help='Path to ddw executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write config without running ddw')

    p.set_defaults(func=_run_refine)
    return p


def _run_refine(args):
    in_dir = Path(args.input)
    sep    = '─' * 70
    prefix = '[DRY RUN] ' if args.dry_run else ''

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # ── Resolve project dir ───────────────────────────────────────────────
    project_dir = args.project_dir
    if project_dir is None:
        proj        = load_or_create()
        project_dir = proj.get('ddw_train', {}).get('output_dir')
        if project_dir:
            print(f'project-dir from project.json: {project_dir}')
        else:
            print('ERROR: --project-dir not specified and no ddw_train in project.json.')
            print('       Run ddw-train first, or pass --project-dir explicitly.')
            sys.exit(1)
    project_dir = Path(project_dir)

    # ── Load train config for mw_angle and subtomo_size ──────────────────
    train_cfg_path = project_dir / 'ddw_config.yaml'
    train_cfg = {}
    if train_cfg_path.exists():
        with open(train_cfg_path) as fh:
            train_cfg = yaml.safe_load(fh) or {}

    mw_angle = args.mw_angle
    if mw_angle is None:
        mw_angle = train_cfg.get('shared', {}).get('mw_angle')
        if mw_angle is not None:
            print(f'Missing wedge angle : {mw_angle}° (from ddw_config.yaml)')
        else:
            mw_angle = _detect_mw_angle(in_dir)
            if mw_angle is not None:
                print(f'Missing wedge angle : {mw_angle}° (auto-detected from .aln)')
            else:
                print('ERROR: could not determine missing wedge angle.')
                print('       Pass --mw-angle explicitly.')
                sys.exit(1)

    subtomo_size = args.subtomo_size
    if subtomo_size is None:
        subtomo_size = train_cfg.get('shared', {}).get('subtomo_size', 96)
        print(f'Subtomogram size    : {subtomo_size} (from ddw_config.yaml)')

    # ── Resolve checkpoint ────────────────────────────────────────────────
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists() and not args.dry_run:
            print(f'ERROR: checkpoint not found: {ckpt_path}')
            sys.exit(1)
    else:
        logdir    = project_dir / 'logs'
        ckpt_path = _find_best_checkpoint(logdir) if logdir.exists() else None
        if ckpt_path is None:
            if args.dry_run:
                ckpt_path = project_dir / 'logs' / '<best_val_loss>.ckpt'
                print(f'Checkpoint          : (will be auto-selected from logs/ after training)')
            else:
                print(f'ERROR: no .ckpt files found in {logdir}/')
                print('       Run ddw-train first, or pass --checkpoint explicitly.')
                sys.exit(1)
        else:
            print(f'Checkpoint          : {ckpt_path.name} (lowest val_loss)')

    # ── Find EVN/ODD pairs ─────────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no EVN/ODD volume pairs found in {in_dir}/')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Apply TS selection filter ──────────────────────────────────────────
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        orig_n = len(pairs)
        pairs  = [p for p in pairs if p['ts_name'] in selected_ts]
        n_excl = orig_n - len(pairs)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(pairs)} remaining')
        if not pairs:
            print('ERROR: no volumes remain after TS selection filter')
            sys.exit(1)

    print(f'Refining {len(pairs)} volumes')
    print(sep)
    for p in pairs[:10]:
        print(f'  {p["ts_name"]}  {p["evn"].name}')
    if len(pairs) > 10:
        print(f'  ... ({len(pairs) - 10} more)')
    print(sep)

    # ── Resolve output dir ────────────────────────────────────────────────
    out_dir = Path(args.output) if args.output else project_dir / 'refined'

    # ── Build refine config YAML ───────────────────────────────────────────
    config = {
        'shared': {
            'project_dir':  str(project_dir.resolve()),
            'tomo0_files':  [str(p['evn'].resolve()) for p in pairs],
            'tomo1_files':  [str(p['odd'].resolve()) for p in pairs],
            'subtomo_size': subtomo_size,
            'mw_angle':     mw_angle,
            'num_workers':  args.num_workers,
            'gpu':          args.gpu,
        },
        'refine_tomogram': {
            'model_checkpoint_file': str(ckpt_path.resolve()),
            'output_dir':            str(out_dir.resolve()),
            'batch_size':            args.batch_size,
            'subtomo_overlap':       subtomo_size // 3,
            'recompute_normalization': True,
        },
    }

    config_path = project_dir / 'ddw_refine_config.yaml'
    print(f'\n{prefix}Config  : {config_path}')
    print(f'{prefix}Output  : {out_dir}/')

    if not args.dry_run:
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(config_path, config)

        if shutil.which(args.ddw_bin) is None and not Path(args.ddw_bin).is_file():
            print(f'ERROR: ddw binary not found: {args.ddw_bin!r}')
            sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────
    print(f'\n{prefix}Step 3: refine-tomogram')
    _run_ddw(args.ddw_bin, 'refine-tomogram', config_path, dry_run=args.dry_run)

    if args.dry_run:
        return

    print(f'\nDeepDeWedge refinement complete.')
    print(f'  Denoised volumes : {out_dir}/')

    update_section(
        section='ddw_refine',
        values={
            'command':    ' '.join(sys.argv),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'n_vols':     len(pairs),
            'checkpoint': str(ckpt_path),
            'output_dir': str(out_dir.resolve()),
        },
        backup_dir=project_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    _add_train_parser(subparsers)
    _add_refine_parser(subparsers)
