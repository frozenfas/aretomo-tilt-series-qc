"""
topaz-denoise3d — denoise reconstructed volumes using Topaz denoise3d.

Applies a denoising model to ts-xxx_Vol.mrc files.  Supports both Topaz
pretrained models and custom .sav checkpoints trained with topaz-train.

Pretrained models
-----------------
  unet-3d      — general-purpose (default if no custom model available)
  unet-3d-10a  — trained on 10 Å/px data
  unet-3d-20a  — trained on 20 Å/px data

Auto-loading a custom model
---------------------------
If --model is omitted and topaz-train has been run, the latest checkpoint
is loaded automatically from project.json (topaz_train.output_dir).
You can also point to the training output dir directly with --project-dir.

Typical usage
-------------
  # Use pretrained model
  aretomo3-preprocess topaz-denoise3d \\
      --input run005-cmd2-sart-thr80 \\
      --output run005-topaz/denoised \\
      --model unet-3d --gpu 0

  # Auto-load custom model from project.json
  aretomo3-preprocess topaz-denoise3d \\
      --input run005-cmd2-sart-thr80 \\
      --output run005-topaz/denoised \\
      --gpu 0

  # Explicit custom model
  aretomo3-preprocess topaz-denoise3d \\
      --input run005-cmd2-sart-thr80 \\
      --output run005-topaz/denoised \\
      --model run005-topaz/model_epoch500.sav --gpu 0
"""

import sys
import subprocess
import shutil
import time
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import load_or_create
from aretomo3_preprocess.commands.topaz_train import _find_latest_model
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_name_from_vol(vol_path: Path, vol_suffix: str) -> str:
    stem = vol_path.stem
    for tag in (
        f'_EVN{vol_suffix}', f'_ODD{vol_suffix}',
        f'{vol_suffix}_EVN', f'{vol_suffix}_ODD',
        vol_suffix,
    ):
        if tag and stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


PRETRAINED_MODELS = {'unet-3d', 'unet-3d-10a', 'unet-3d-20a'}


def _resolve_model(args) -> str:
    """
    Resolve the model to use, in priority order:
      1. --model  (explicit pretrained name or .sav path)
      2. --project-dir  → latest .sav in that directory
      3. project.json → topaz_train.output_dir → latest .sav
      4. Fall back to 'unet-3d' pretrained

    Returns the model string to pass to topaz -m.
    """
    # 1. Explicit --model
    if args.model is not None:
        return args.model

    # 2. --project-dir given
    project_dir = getattr(args, 'project_dir', None)
    if project_dir is not None:
        project_dir = Path(project_dir)
        model = _find_latest_model(project_dir)
        if model:
            print(f'Model               : {model} (latest checkpoint in --project-dir)')
            return str(model)
        print(f'WARNING: no .sav checkpoints found in {project_dir}/')

    # 3. Auto-load from project.json
    if project_dir is None:
        proj        = load_or_create()
        output_dir  = proj.get('topaz_train', {}).get('output_dir')
        if output_dir:
            output_dir = Path(output_dir)
            model = _find_latest_model(output_dir)
            if model:
                print(f'Model               : {model}')
                print(f'                      (latest checkpoint from project.json topaz_train)')
                return str(model)
            else:
                print(f'WARNING: topaz_train.output_dir in project.json is {output_dir}')
                print(f'         but no .sav checkpoints found there.')

    # 4. Pretrained fallback
    print('Model               : unet-3d (pretrained fallback — no custom model found)')
    return 'unet-3d'


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'topaz-denoise3d',
        help='Denoise reconstructed volumes with Topaz denoise3d',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-xxx_Vol.mrc files')
    inp.add_argument('--vol-suffix', default='_Vol',
                     help='Volume filename suffix (default: _Vol → ts-xxx_Vol.mrc)')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only selected TS are denoised')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default=None,
                     help='Output directory (default: <input>/topaz_denoise)')

    mdl = p.add_argument_group('model')
    mdl.add_argument('--model', '-m', default=None,
                     metavar='MODEL_OR_PATH',
                     help='Pretrained model name (unet-3d / unet-3d-10a / unet-3d-20a) '
                          'or path to a custom .sav checkpoint from topaz-train. '
                          'Default: auto-load from project.json, then unet-3d.')
    mdl.add_argument('--project-dir', default=None, metavar='DIR',
                     help='Topaz training output directory; scan for latest .sav '
                          'checkpoint. If omitted, auto-loaded from project.json.')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML QC report with side-by-side before/after '
                         'central-slab projections for each denoised volume')
    qc.add_argument('--analyse-thickness', type=float, default=300.0, metavar='ANGST',
                    help='Slab thickness in Å for QC projections (default: 300 Å)')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/topaz_denoise_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', '-d', type=int, default=0,
                     help='GPU device index (default: 0; use -1 for CPU)')
    ctl.add_argument('--patch-size', type=int, default=96,
                     help='Patch size for denoising (default: 96)')
    ctl.add_argument('--patch-padding', type=int, default=48,
                     help='Patch padding to reduce edge artefacts (default: 48)')
    ctl.add_argument('--topaz', dest='topaz_bin',
                     default='/opt/miniconda3/envs/topaz/bin/topaz',
                     help='Path to topaz executable')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without executing')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input)
    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} is not a directory')
        sys.exit(1)

    out_dir = Path(args.output) if args.output else in_dir / 'topaz_denoise'
    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)
    sfx     = args.vol_suffix

    summary = []
    if args.dry_run and Path(out_dir).exists():
        summary.append(f'NOTE: output directory already exists: {out_dir}')
        summary.append(f'      (would prompt or require --clean in a real run)')

    # ── Resolve model ─────────────────────────────────────────────────────────
    model = _resolve_model(args)

    # ── TS selection ──────────────────────────────────────────────────────────
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))

    # ── Find volumes ──────────────────────────────────────────────────────────
    vol_files = sorted(in_dir.glob(f'ts-*{sfx}.mrc'))

    # Exclude EVN/ODD half-sets
    vol_files = [
        f for f in vol_files
        if not (f.stem.endswith(f'_EVN{sfx}') or f.stem.endswith(f'_ODD{sfx}')
                or '_EVN' in f.stem or '_ODD' in f.stem)
    ]

    if not vol_files:
        print(f'ERROR: no volumes found in {in_dir}/ matching ts-*{sfx}.mrc')
        sys.exit(1)

    if selected_ts is not None:
        before    = len(vol_files)
        vol_files = [f for f in vol_files
                     if _ts_name_from_vol(f, sfx) in selected_ts]
        summary.append(f'TS selection        : {len(vol_files)} / {before} volumes retained')

    if not vol_files:
        print('No volumes to process after selection filter.')
        sys.exit(0)

    n   = len(vol_files)
    w   = len(str(n))
    sep = '─' * 70

    is_pretrained = model in PRETRAINED_MODELS
    model_label   = model if is_pretrained else Path(model).name

    summary.append(f'Volumes to process  : {n}')
    summary.append(f'Model               : {model_label}' + (' (pretrained)' if is_pretrained else ''))
    summary.append(f'Patch size / pad    : {args.patch_size} / {args.patch_padding}')
    summary.append(f'Device              : GPU {args.gpu}')
    summary.append(f'Output directory    : {out_dir}/')

    if not args.dry_run:
        for line in summary:
            print(line)
        print()

    # ── Check binary ──────────────────────────────────────────────────────────
    if not args.dry_run:
        if (shutil.which(args.topaz_bin) is None
                and not Path(args.topaz_bin).is_file()):
            print(f'ERROR: topaz binary not found: {args.topaz_bin!r}')
            print('       Install topaz or use --topaz /path/to/topaz')
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── QC setup ──────────────────────────────────────────────────────────────
    do_qc    = getattr(args, 'analyse', False)
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    if do_qc:
        try:
            from aretomo3_preprocess.shared.volume_qc import (
                central_slab_projection, projection_to_b64png, make_comparison_html,
            )
        except ImportError:
            print('WARNING: --analyse requires mrcfile and matplotlib; skipping report')
            do_qc = False

    # ── Process ───────────────────────────────────────────────────────────────
    prefix   = '[DRY RUN] ' if args.dry_run else ''
    n_ok     = n_fail = 0
    t_start  = time.perf_counter()

    for i, vol_path in enumerate(vol_files, 1):
        ts_name  = _ts_name_from_vol(vol_path, sfx)
        out_path = out_dir / vol_path.name

        cmd = [
            args.topaz_bin, 'denoise3d',
            str(vol_path),
            '-o',  str(out_dir),
            '-m',  model,
            '-d',  str(args.gpu),
            '-s',  str(args.patch_size),
            '-p',  str(args.patch_padding),
        ]

        print(sep, flush=True)
        print(f'{prefix}[{i:{w}d}/{n}]  {vol_path.name}', flush=True)

        if args.dry_run:
            print(f'  cmd: {" ".join(cmd)}')
            n_ok += 1
            continue

        # Capture before projection
        before_b64 = None
        if do_qc:
            proj = central_slab_projection(vol_path, qc_thick)
            if proj:
                before_b64 = projection_to_b64png(proj['img'])

        t0     = time.perf_counter()
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f'  FAILED  (exit {result.returncode}, {elapsed:.0f}s)', flush=True)
            for line in (result.stderr or '').strip().splitlines():
                print(f'    {line}')
            n_fail += 1
            after_b64 = None
        else:
            elapsed_total = time.perf_counter() - t_start
            rate = i / elapsed_total
            eta  = (n - i) / rate if rate > 0 else 0
            print(f'  done  ({elapsed:.0f}s  |  {i}/{n} complete'
                  f'  |  ETA {eta/60:.0f} min)', flush=True)
            n_ok += 1
            after_b64 = None
            if do_qc and out_path.exists():
                proj = central_slab_projection(out_path, qc_thick)
                if proj:
                    after_b64 = projection_to_b64png(proj['img'])

        if do_qc:
            qc_entries.append({
                'ts_name':     ts_name,
                'before_b64':  before_b64,
                'after_b64':   after_b64,
                'before_path': str(vol_path),
                'after_path':  str(out_path),
                'metadata': {
                    'model':         model_label,
                    'patch size':    str(args.patch_size),
                    'patch padding': str(args.patch_padding),
                },
            })

    print(sep)
    total = time.perf_counter() - t_start
    print(f'\nDone: {n_ok} denoised, {n_fail} failed  '
          f'(total {total/60:.1f} min)')

    if args.dry_run and summary:
        print()
        print('── Summary ─────────────────────────────────────────────────────────')
        for line in summary:
            print(line)

    # ── QC report ─────────────────────────────────────────────────────────────
    if do_qc and qc_entries:
        html_path = (Path(args.analyse_output) if args.analyse_output
                     else out_dir / 'topaz_denoise_qc.html')
        make_comparison_html(
            entries      = qc_entries,
            out_path     = html_path,
            title        = 'topaz-denoise3d QC',
            command      = ' '.join(sys.argv),
            before_label = 'Before (raw)',
            after_label  = 'After (denoised)',
            slab_angst   = qc_thick,
        )

    if n_fail:
        sys.exit(1)
