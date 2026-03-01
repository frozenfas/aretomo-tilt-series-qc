"""
cryocare-train  — extract training data and train a cryoCARE denoising model.
cryocare-predict — denoise tomograms using a trained cryoCARE model.

cryoCARE requires even/odd half-set volumes produced by AreTomo3 with
-SplitSum 1 (--split-sum 1).  EVN/ODD pairs are discovered by globbing
  ts-*{--vol-suffix}_EVN.mrc / ts-*{--vol-suffix}_ODD.mrc
in the input directory (--vol-suffix defaults to '', matching ts-xxx_EVN.mrc).
For multi-resolution runs (--at-bin 4 8), use e.g. --vol-suffix '_b4' to select
the bin-4 output files.

For training, a stratified random subset of N volumes is selected to span the
defocus range of the dataset.  Defocus is read from alignment_data.json in
--analysis (using the acq_order=1 image — lowest/zero tilt, most accurate
CTF estimate).  Volumes with too few aligned tilts can be excluded with
--min-tilts.  The stratification divides the defocus range into N equal bins
and draws one volume at random from each bin.

The commands write JSON config files and call the cryoCARE scripts:
  cryoCARE_extract_train_data.py   (train only — step 1)
  cryoCARE_train.py                (train only — step 2)
  cryoCARE_predict.py              (predict only)

Scripts must be on PATH (load the module first) or use --cryocare-dir:
  module load cryoCARE/0.3
  aretomo3-preprocess cryocare-train ...

Typical usage
-------------
  # Train on bin-4 volumes from run004, using analysis QC filter
  aretomo3-preprocess cryocare-train \\
      --input run004 --vol-suffix _b4 \\
      --analysis run001_analysis \\
      --n-vols 8 --min-tilts 20 \\
      --gpu 0 --output cryocare_train --dry-run

  # Predict on all bin-4 volumes using the trained model
  aretomo3-preprocess cryocare-predict \\
      --input run004 --vol-suffix _b4 \\
      --output run004_denoised \\
      --gpu 0 --dry-run
"""

import sys
import json
import shutil
import random
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import (
    load_or_create, update_section, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import get_latest_analysis_dir


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_evn_odd_pairs(in_dir, vol_suffix):
    """
    Return sorted list of dicts {ts_name, evn, odd} for all EVN/ODD pairs in
    in_dir matching ts-*{vol_suffix}_EVN.mrc / ts-*{vol_suffix}_ODD.mrc.
    """
    in_dir = Path(in_dir)
    suffix_tag = f'{vol_suffix}_EVN'   # what the stem ends with
    pairs = []
    for evn in sorted(in_dir.glob(f'ts-*{vol_suffix}_EVN.mrc')):
        if not evn.stem.endswith(suffix_tag):
            continue
        ts_name = evn.stem[: -len(suffix_tag)]
        odd = in_dir / f'{ts_name}{vol_suffix}_ODD.mrc'
        if odd.exists():
            pairs.append({'ts_name': ts_name, 'evn': evn, 'odd': odd})
        else:
            print(f'Warning: {evn.name} has no matching ODD — skipping')
    return pairs


def _load_analysis(analysis_dir, ts_names):
    """
    Load per-TS defocus (acq_order=1 image) and n_tilts (aligned frames) from
    alignment_data.json in analysis_dir.

    Returns dict: ts_name → {'defocus_um': float|None, 'n_tilts': int}
    """
    result = {ts: {'defocus_um': None, 'n_tilts': 0} for ts in ts_names}
    if analysis_dir is None:
        return result

    aln_path = Path(analysis_dir) / 'alignment_data.json'
    if not aln_path.exists():
        print(f'Warning: {aln_path} not found — defocus/n_tilts unavailable')
        return result

    with open(aln_path) as fh:
        aln = json.load(fh)

    for ts in ts_names:
        ts_data = aln.get(ts, {})
        frames  = ts_data.get('frames', [])
        n_tilts = len(frames)
        defocus_um = None
        for f in frames:
            if f.get('acq_order') == 1:
                defocus_um = f.get('mean_defocus_um')
                break
        result[ts] = {'defocus_um': defocus_um, 'n_tilts': n_tilts}

    return result


def _stratified_sample(candidates, n, seed=42):
    """
    Stratified random sample of n ts_names from candidates, based on defocus_um.

    candidates: list of dicts with ts_name and defocus_um keys.
    Divides the defocus range into n equal bins and picks one per bin.
    Items without defocus are pooled and sampled last to make up any shortfall.
    Returns sorted list of selected ts_names.
    """
    if n >= len(candidates):
        return sorted(c['ts_name'] for c in candidates)

    rng = random.Random(seed)

    with_def = sorted(
        [c for c in candidates if c.get('defocus_um') is not None],
        key=lambda c: c['defocus_um'],
    )
    no_def = [c for c in candidates if c.get('defocus_um') is None]

    selected = []
    n_strat = min(n, len(with_def))
    if n_strat > 0:
        total    = len(with_def)
        bin_size = total / n_strat
        for i in range(n_strat):
            lo  = int(i * bin_size)
            hi  = max(lo + 1, int((i + 1) * bin_size))
            pick = rng.choice(with_def[lo:hi])
            selected.append(pick['ts_name'])

    n_extra = n - len(selected)
    if n_extra > 0 and no_def:
        extras = rng.sample(no_def, min(n_extra, len(no_def)))
        selected.extend(c['ts_name'] for c in extras)

    return sorted(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run argument table
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_ARG_HELP = {
    'input':                   'directory containing EVN/ODD half-set volumes',
    'vol_suffix':              'suffix between ts-name and _EVN/_ODD (e.g. "_b4")',
    'analysis':                'analyse output dir for defocus/n_tilts data',
    'n_vols':                  'volumes for training, selected stratified by defocus',
    'min_tilts':               'exclude volumes with fewer than N aligned tilts',
    'output':                  'output dir for configs, training data and model',
    'model_name':              'model file stem (saved as <model_name>.tar.gz)',
    'patch_shape':             'training patch size in voxels [X Y Z]',
    'num_slices':              'sub-volume patches extracted per tomogram',
    'tilt_axis':               'tilt axis in volume (0=X 1=Y 2=Z); 1=Y for AreTomo3 FlipVol=1',
    'n_normalization_samples': 'patches used to estimate normalisation statistics',
    'epochs':                  'training epochs',
    'steps_per_epoch':         'gradient steps per epoch',
    'batch_size':              'training mini-batch size',
    'unet_kern_size':          'U-Net convolution kernel size',
    'unet_n_depth':            'U-Net depth (number of down/up levels)',
    'unet_n_first':            'U-Net first-layer feature maps',
    'learning_rate':           'Adam optimiser learning rate',
    'gpu':                     'GPU device ID',
    'cryocare_dir':            'explicit path to cryoCARE scripts (if not on PATH)',
    'dry_run':                 'write configs without running cryoCARE',
}

_PREDICT_ARG_HELP = {
    'input':       'directory containing EVN/ODD half-set volumes',
    'vol_suffix':  'suffix between ts-name and _EVN/_ODD (e.g. "_b4")',
    'model':       'trained model .tar.gz (or from project.json if omitted)',
    'analysis':    'analyse output dir for --min-tilts filtering',
    'min_tilts':   'exclude volumes with fewer than N aligned tilts',
    'output':      'output dir for denoised volumes',
    'overwrite':   'overwrite existing predictions',
    'n_tiles':     'prediction tiling [X Y Z] (reduce if GPU runs out of memory)',
    'gpu':         'GPU device ID',
    'cryocare_dir': 'explicit path to cryoCARE scripts (if not on PATH)',
    'dry_run':     'write predict_config.json without running cryoCARE',
}


def _print_args_table(args, descriptions):
    """Print a formatted argument table for dry-run."""
    skip = {'func', 'command'}
    items = [(k, v) for k, v in sorted(vars(args).items()) if k not in skip]
    w_key = max(len(k) for k, _ in items)
    w_val = min(30, max(len(str(v)) for _, v in items))
    print('── Arguments ───────────────────────────────────────────────────────')
    for k, v in items:
        desc = descriptions.get(k, '')
        print(f'  {k:<{w_key}}  {str(v):<{w_val}}  {desc}')
    print()


def _find_script(name, cryocare_dir=None):
    """Return the path to a cryoCARE script, or None if not found."""
    if cryocare_dir is not None:
        p = Path(cryocare_dir) / name
        if p.exists():
            return str(p)
    return shutil.which(name)


def _run_script(script, conf_path, dry_run):
    """Run `script --conf conf_path`, streaming stdout live."""
    cmd = [script, '--conf', str(conf_path)]
    print(f'  $ {" ".join(cmd)}\n')
    if dry_run:
        return
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f'ERROR: script exited with code {ret.returncode}')
        sys.exit(ret.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# cryocare-train
# ─────────────────────────────────────────────────────────────────────────────

def _add_train_parser(subparsers):
    p = subparsers.add_parser(
        'cryocare-train',
        help='Train a cryoCARE denoising model on EVN/ODD half-set volumes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_EVN.mrc / _ODD.mrc volumes')
    inp.add_argument('--vol-suffix', default='',
                     help='Suffix between ts-name and _EVN/_ODD '
                          '(e.g. "_b4" for ts-xxx_b4_EVN.mrc)')
    inp.add_argument('--analysis', default=None,
                     help='analyse output dir (alignment_data.json) for '
                          'defocus-based stratification and --min-tilts filter')

    filt = p.add_argument_group('volume selection')
    filt.add_argument('--n-vols', type=int, default=8,
                      help='Number of volumes for training (stratified by defocus)')
    filt.add_argument('--min-tilts', type=int, default=None,
                      help='Exclude volumes with fewer than N aligned tilts')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default='cryocare_train',
                     help='Output directory for configs, training data and model')
    out.add_argument('--model-name', default='cryocare_model',
                     help='Model name (subdirectory/file stem under --output)')

    td = p.add_argument_group('training data extraction')
    td.add_argument('--patch-shape', type=int, nargs=3, default=[72, 72, 72],
                    metavar=('X', 'Y', 'Z'),
                    help='Patch shape in voxels')
    td.add_argument('--num-slices', type=int, default=1200,
                    help='Sub-volume patches per tomogram')
    td.add_argument('--tilt-axis', type=int, default=1,
                    help='Tilt axis in the volume (0=X 1=Y 2=Z); '
                         '1 is correct for AreTomo3 with FlipVol=1')
    td.add_argument('--n-normalization-samples', type=int, default=500,
                    help='Patches used to estimate normalisation statistics')

    tr = p.add_argument_group('training')
    tr.add_argument('--epochs', type=int, default=100,
                    help='Training epochs')
    tr.add_argument('--steps-per-epoch', type=int, default=200,
                    help='Steps per epoch')
    tr.add_argument('--batch-size', type=int, default=16,
                    help='Batch size')
    tr.add_argument('--unet-kern-size', type=int, default=3,
                    help='U-Net kernel size')
    tr.add_argument('--unet-n-depth', type=int, default=3,
                    help='U-Net depth (number of down/up levels)')
    tr.add_argument('--unet-n-first', type=int, default=32,
                    help='U-Net first-layer feature maps')
    tr.add_argument('--learning-rate', type=float, default=0.0004,
                    help='Learning rate')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, default=0,
                     help='GPU ID')
    ctl.add_argument('--cryocare-dir', default=None,
                     help='Directory containing cryoCARE scripts '
                          '(omit if module is already loaded)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write config files without running cryoCARE')

    p.set_defaults(func=_run_train)
    return p


def _run_train(args):
    # Auto-fill --analysis from project.json if not given
    if args.analysis is None:
        _auto = get_latest_analysis_dir()
        if _auto is not None:
            args.analysis = str(_auto)
            print(f'Note: --analysis not given; using project.json → {args.analysis}')

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    sep     = '─' * 70

    if args.dry_run:
        _print_args_table(args, _TRAIN_ARG_HELP)

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # ── Discover EVN/ODD pairs ─────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no ts-*{args.vol_suffix}_EVN.mrc / _ODD.mrc pairs '
              f'found in {in_dir}/')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Load analysis data ─────────────────────────────────────────────────
    ts_names = [p['ts_name'] for p in pairs]
    ana_data = _load_analysis(args.analysis, ts_names)

    # ── Filter by --min-tilts ──────────────────────────────────────────────
    candidates = []
    n_excluded = 0
    for pair in pairs:
        ts   = pair['ts_name']
        info = ana_data[ts]
        if args.min_tilts is not None and info['n_tilts'] < args.min_tilts:
            n_excluded += 1
            continue
        candidates.append({
            'ts_name':    ts,
            'evn':        pair['evn'],
            'odd':        pair['odd'],
            'defocus_um': info['defocus_um'],
            'n_tilts':    info['n_tilts'],
        })

    if n_excluded:
        print(f'Excluded {n_excluded} volumes with < {args.min_tilts} aligned tilts')
    if not candidates:
        print('ERROR: no volumes remain after filtering')
        sys.exit(1)

    # ── Stratified selection ───────────────────────────────────────────────
    selected_names = _stratified_sample(candidates, args.n_vols)
    selected = [c for c in candidates if c['ts_name'] in set(selected_names)]

    print(f'\nTraining volumes: {len(selected)} selected '
          f'(stratified by defocus, from {len(candidates)} candidates)')
    print(sep)
    for c in selected:
        def_str = f'{c["defocus_um"]:.2f} μm' if c['defocus_um'] is not None else 'unknown'
        tlt_str = str(c['n_tilts']) if c['n_tilts'] else '?'
        print(f'  {c["ts_name"]:12s}  defocus={def_str:12s}  n_tilts={tlt_str}')
    print(sep)

    # ── Write config files ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = out_dir / 'train_data'

    train_data_config = {
        'even':                    [str(c['evn'].resolve()) for c in selected],
        'odd':                     [str(c['odd'].resolve()) for c in selected],
        'patch_shape':             args.patch_shape,
        'num_slices':              args.num_slices,
        'tilt_axis':               args.tilt_axis,
        'n_normalization_samples': args.n_normalization_samples,
        'path':                    str(patches_dir.resolve()),
    }

    train_config = {
        'train_data':      str(patches_dir.resolve()),
        'epochs':          args.epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'batch_size':      args.batch_size,
        'unet_kern_size':  args.unet_kern_size,
        'unet_n_depth':    args.unet_n_depth,
        'unet_n_first':    args.unet_n_first,
        'learning_rate':   args.learning_rate,
        'model_name':      args.model_name,
        'path':            str(out_dir.resolve()),
        'gpu_id':          args.gpu,
    }

    td_conf = out_dir / 'train_data_config.json'
    tr_conf = out_dir / 'train_config.json'

    with open(td_conf, 'w') as fh:
        json.dump(train_data_config, fh, indent=2)
    with open(tr_conf, 'w') as fh:
        json.dump(train_config, fh, indent=2)

    prefix = '[DRY RUN] ' if args.dry_run else ''
    print(f'\n{prefix}Config files:')
    print(f'  {td_conf}')
    print(f'  {tr_conf}')

    # ── Locate cryoCARE scripts ────────────────────────────────────────────
    ccd = args.cryocare_dir
    ext_script = _find_script('cryoCARE_extract_train_data.py', ccd)
    trn_script = _find_script('cryoCARE_train.py', ccd)

    for name, found in [('cryoCARE_extract_train_data.py', ext_script),
                        ('cryoCARE_train.py', trn_script)]:
        if not found:
            msg = f'WARNING: {name} not found on PATH'
            if args.dry_run:
                print(f'\n{msg} (dry-run: ignoring)')
            else:
                print(f'\nERROR: {msg}')
                print('  Load the module first: module load cryoCARE/0.3')
                print('  Or specify: --cryocare-dir /path/to/cryocare/bin')
                sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────
    print(f'\n{prefix}Step 1: Extract training data')
    _run_script(ext_script or 'cryoCARE_extract_train_data.py',
                td_conf, args.dry_run)

    print(f'{prefix}Step 2: Train cryoCARE model')
    _run_script(trn_script or 'cryoCARE_train.py',
                tr_conf, args.dry_run)

    if args.dry_run:
        return

    model_path = out_dir / f'{args.model_name}.tar.gz'
    print(f'\ncryoCARE training complete.')
    if model_path.exists():
        print(f'Model: {model_path}')
    else:
        # cryoCARE may save the model differently — scan for it
        tar_files = list(out_dir.glob('*.tar.gz'))
        if tar_files:
            model_path = tar_files[0]
            print(f'Model: {model_path}')
        else:
            print(f'Model: see {out_dir}/ (expected {model_path.name})')

    update_section(
        section='cryocare_train',
        values={
            'command':    ' '.join(sys.argv),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'n_vols':     len(selected),
            'selected':   [c['ts_name'] for c in selected],
            'model_path': str(model_path),
            'output_dir': str(out_dir.resolve()),
        },
        backup_dir=out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# cryocare-predict
# ─────────────────────────────────────────────────────────────────────────────

def _add_predict_parser(subparsers):
    p = subparsers.add_parser(
        'cryocare-predict',
        help='Denoise tomograms with a trained cryoCARE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_EVN.mrc / _ODD.mrc volumes')
    inp.add_argument('--vol-suffix', default='',
                     help='Suffix between ts-name and _EVN/_ODD '
                          '(e.g. "_b4" for ts-xxx_b4_EVN.mrc)')
    inp.add_argument('--model', '-m', default=None,
                     help='Path to trained model (.tar.gz). '
                          'If omitted, reads model_path from project.json '
                          '[cryocare_train].')
    inp.add_argument('--analysis', default=None,
                     help='analyse output dir for --min-tilts filtering')

    filt = p.add_argument_group('volume selection')
    filt.add_argument('--min-tilts', type=int, default=None,
                      help='Exclude volumes with fewer than N aligned tilts')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default='cryocare_predict',
                     help='Output directory for denoised volumes')
    out.add_argument('--overwrite', action='store_true', default=True,
                     help='Overwrite existing predictions')

    pred = p.add_argument_group('prediction')
    pred.add_argument('--n-tiles', type=int, nargs=3, default=[4, 4, 4],
                      metavar=('X', 'Y', 'Z'),
                      help='Prediction tiling (reduce if GPU OOM)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, default=0,
                     help='GPU ID')
    ctl.add_argument('--cryocare-dir', default=None,
                     help='Directory containing cryoCARE scripts '
                          '(omit if module is already loaded)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Write predict_config.json without running cryoCARE')

    p.set_defaults(func=_run_predict)
    return p


def _run_predict(args):
    # Auto-fill --analysis from project.json if not given
    if args.analysis is None:
        _auto = get_latest_analysis_dir()
        if _auto is not None:
            args.analysis = str(_auto)

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    sep     = '─' * 70

    if args.dry_run:
        _print_args_table(args, _PREDICT_ARG_HELP)

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    # ── Resolve model path ─────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        proj = load_or_create()
        model_path = proj.get('cryocare_train', {}).get('model_path')
        if model_path:
            print(f'Using model from project.json: {model_path}')
        else:
            print('ERROR: --model not specified and no cryocare_train in '
                  'project.json.\n'
                  '       Run cryocare-train first, or pass --model '
                  '/path/to/model.tar.gz')
            sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f'ERROR: model file not found: {model_path}')
        sys.exit(1)

    # ── Discover EVN/ODD pairs ─────────────────────────────────────────────
    pairs = _find_evn_odd_pairs(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no ts-*{args.vol_suffix}_EVN.mrc / _ODD.mrc pairs '
              f'found in {in_dir}/')
        sys.exit(1)
    print(f'Found {len(pairs)} EVN/ODD pairs in {in_dir}/')

    # ── Load analysis data and filter ─────────────────────────────────────
    ts_names = [p['ts_name'] for p in pairs]
    ana_data = _load_analysis(args.analysis, ts_names)

    selected = []
    n_excluded = 0
    for pair in pairs:
        ts   = pair['ts_name']
        info = ana_data[ts]
        if args.min_tilts is not None and info['n_tilts'] < args.min_tilts:
            n_excluded += 1
            continue
        selected.append(pair)

    if n_excluded:
        print(f'Excluded {n_excluded} volumes with < {args.min_tilts} aligned tilts')

    if not selected:
        print('ERROR: no volumes remain after filtering')
        sys.exit(1)

    print(f'Predicting on {len(selected)} volumes')
    print(sep)
    for p in selected[:10]:
        print(f'  {p["ts_name"]}')
    if len(selected) > 10:
        print(f'  ... ({len(selected) - 10} more)')
    print(sep)

    # ── Write predict config ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_config = {
        'path':      str(model_path.resolve()),
        'even':      [str(p['evn'].resolve()) for p in selected],
        'odd':       [str(p['odd'].resolve()) for p in selected],
        'n_tiles':   args.n_tiles,
        'output':    str(out_dir.resolve()),
        'overwrite': args.overwrite,
        'gpu_id':    args.gpu,
    }

    pred_conf = out_dir / 'predict_config.json'
    with open(pred_conf, 'w') as fh:
        json.dump(predict_config, fh, indent=2)

    prefix = '[DRY RUN] ' if args.dry_run else ''
    print(f'\n{prefix}Config file:')
    print(f'  {pred_conf}')

    # ── Locate cryoCARE predict script ─────────────────────────────────────
    pred_script = _find_script('cryoCARE_predict.py', args.cryocare_dir)
    if not pred_script:
        msg = 'WARNING: cryoCARE_predict.py not found on PATH'
        if args.dry_run:
            print(f'\n{msg} (dry-run: ignoring)')
        else:
            print(f'\nERROR: {msg}')
            print('  Load the module first: module load cryoCARE/0.3')
            print('  Or specify: --cryocare-dir /path/to/cryocare/bin')
            sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────
    print(f'\n{prefix}Running cryoCARE prediction:')
    _run_script(pred_script or 'cryoCARE_predict.py', pred_conf, args.dry_run)

    if args.dry_run:
        return

    print(f'\ncryoCARE prediction complete.  Denoised volumes in {out_dir}/')

    update_section(
        section='cryocare_predict',
        values={
            'command':    ' '.join(sys.argv),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'n_vols':     len(selected),
            'model_path': str(model_path),
            'output_dir': str(out_dir.resolve()),
        },
        backup_dir=out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    _add_train_parser(subparsers)
    _add_predict_parser(subparsers)
