"""
topaz-train — train a Topaz denoise3d model on EVN/ODD half-set volumes.

Uses the noise2noise training built into `topaz denoise3d` to fit a custom
denoising model on your own data.  EVN/ODD volumes must be produced by
AreTomo3 with -SplitSum 1 (--split-sum 1).

Topaz's -a/-b arguments accept a directory of MRC files with matching names.
This command creates evn/ and odd/ staging subdirectories under --output with
symlinks named ts-xxx.mrc pointing to the respective half-set volumes, then
calls topaz with those directories.

After training, use topaz-denoise3d with --model <output>/model_epoch<N>.sav
to apply the trained model, or omit --model to auto-load from project.json.

GPU device convention (--gpu):
  >= 0   : single GPU (e.g. --gpu 0)
     -2  : all available GPUs (DataParallel, default)
     -1  : CPU

Typical usage
-------------
  # Train on selected volumes
  aretomo3-preprocess topaz-train \\
      --input run005-cmd2-sart-thr80/ \\
      --output run005-topaz/ \\
      --select-ts ts-select-20tilts-thr80-tilt865-run002.csv \\
      --gpu -2 \\
      --dry-run

  # Denoise using the trained model (auto-loaded from project.json)
  aretomo3-preprocess topaz-denoise3d \\
      --input run005-cmd2-sart-thr80/ \\
      --output run005-topaz/denoised/ \\
      --gpu 0
"""

import sys
import shutil
import struct
import datetime
import subprocess
import time
from pathlib import Path
import argparse

import numpy as np

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
# MRC I/O helpers (no mrcfile dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _read_mrc(path: Path) -> np.ndarray:
    """Read a float32 MRC volume as a (Z, Y, X) numpy array."""
    with open(path, 'rb') as f:
        hdr = f.read(1024)
        nx, ny, nz = struct.unpack_from('<3i', hdr, 0)
        mode = struct.unpack_from('<i', hdr, 12)[0]
        nsymbt = struct.unpack_from('<i', hdr, 92)[0]  # extra header bytes
        dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
        dtype = dtype_map.get(mode, np.float32)
        f.seek(1024 + nsymbt)
        data = np.frombuffer(f.read(), dtype=dtype).reshape(nz, ny, nx)
    return data.astype(np.float32)


def _write_mrc_float32(path: Path, data: np.ndarray):
    """Write a float32 3D (Z, Y, X) array as an MRC2014 file."""
    assert data.ndim == 3
    data = data.astype(np.float32)
    nz, ny, nx = data.shape
    hdr = bytearray(1024)
    struct.pack_into('<3i', hdr,  0, nx, ny, nz)          # NX NY NZ
    struct.pack_into('<i',  hdr, 12, 2)                    # MODE 2 = float32
    struct.pack_into('<3i', hdr, 28, nx, ny, nz)           # MX MY MZ
    struct.pack_into('<3f', hdr, 40, float(nx), float(ny), float(nz))  # CELLA
    struct.pack_into('<3f', hdr, 52, 90.0, 90.0, 90.0)    # CELLB angles
    struct.pack_into('<3i', hdr, 64, 1, 2, 3)              # MAPC MAPR MAPS
    struct.pack_into('<3f', hdr, 76, float(data.min()),
                     float(data.max()), float(data.mean()))  # DMIN DMAX DMEAN
    struct.pack_into('<i',  hdr, 88, 1)                    # ISPG = 1
    hdr[208:212] = b'MAP '
    hdr[212:216] = bytes([0x44, 0x44, 0x00, 0x00])        # MACHST little-endian
    struct.pack_into('<f',  hdr, 216, float(np.std(data))) # RMS
    with open(path, 'wb') as f:
        f.write(bytes(hdr))
        f.write(data.tobytes())


def _bin3d(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Block-average a 3D (Z, Y, X) float32 array by integer factor.
    Trailing voxels that don't fit a complete block are dropped.
    """
    nz, ny, nx = arr.shape
    nz2 = nz // factor
    ny2 = ny // factor
    nx2 = nx // factor
    return (arr[:nz2 * factor, :ny2 * factor, :nx2 * factor]
            .reshape(nz2, factor, ny2, factor, nx2, factor)
            .mean(axis=(1, 3, 5))
            .astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_model(model_dir: Path):
    """
    Find the most recent .sav checkpoint in model_dir.
    Prefers the highest epoch number in the filename; falls back to mtime.
    """
    sav_files = list(model_dir.glob('*.sav'))
    if not sav_files:
        return None

    import re
    epoch_files = []
    for f in sav_files:
        m = re.search(r'epoch(\d+)', f.name)
        if m:
            epoch_files.append((int(m.group(1)), f))

    if epoch_files:
        return sorted(epoch_files, key=lambda x: x[0])[-1][1]
    return sorted(sav_files, key=lambda p: p.stat().st_mtime)[-1]


def _stage_training_dirs(out_dir: Path, pairs: list,
                         dry_run: bool = False) -> tuple:
    """
    Create <out_dir>/training_data/evn/ and .../odd/ with matching symlinks.

    Topaz requires both -a and -b to be directories with identical filenames.
    We create ts-xxx.mrc symlinks in each half so the names match.

    Returns (evn_dir, odd_dir).
    """
    td   = out_dir / 'training_data'
    evn_dir = td / 'evn'
    odd_dir = td / 'odd'

    prefix = '[DRY RUN] ' if dry_run else ''
    print(f'{prefix}Staging training dirs:')
    print(f'  {evn_dir}/')
    print(f'  {odd_dir}/')

    if dry_run:
        return evn_dir, odd_dir

    evn_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        ts   = pair['ts_name']
        name = f'{ts}.mrc'
        evn_link = evn_dir / name
        odd_link = odd_dir / name
        if not evn_link.exists() and not evn_link.is_symlink():
            evn_link.symlink_to(pair['evn'].resolve())
        if not odd_link.exists() and not odd_link.is_symlink():
            odd_link.symlink_to(pair['odd'].resolve())

    return evn_dir, odd_dir


def _stage_binned_dirs(out_dir: Path, pairs: list,
                       bin_factor: int, dry_run: bool = False) -> tuple:
    """
    Create binned MRC copies in <out_dir>/training_data/evn_b{N}/ and .../odd_b{N}/.

    Files are written as float32 MRC.  Already-existing files are skipped
    (safe to re-run).  Returns (evn_dir, odd_dir).
    """
    tag = f'b{bin_factor}'
    td = out_dir / 'training_data'
    evn_dir = td / f'evn_{tag}'
    odd_dir = td / f'odd_{tag}'

    prefix = '[DRY RUN] ' if dry_run else ''
    print(f'{prefix}Staging bin-{bin_factor} training dirs:')
    print(f'  {evn_dir}/')
    print(f'  {odd_dir}/')

    if dry_run:
        return evn_dir, odd_dir

    evn_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        name = f'{pair["ts_name"]}.mrc'
        for src_path, dst_dir in [(pair['evn'], evn_dir), (pair['odd'], odd_dir)]:
            dst = dst_dir / name
            if dst.exists():
                continue
            print(f'  Binning {src_path.name} → {dst.name} ... ', end='', flush=True)
            vol = _read_mrc(src_path.resolve())
            binned = _bin3d(vol, bin_factor)
            _write_mrc_float32(dst, binned)
            print(f'{vol.shape} → {binned.shape}')

    return evn_dir, odd_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'topaz-train',
        help='Train a Topaz denoise3d model on EVN/ODD half-set volumes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_EVN_Vol.mrc / _ODD_Vol.mrc')
    inp.add_argument('--vol-suffix', default='',
                     help='Volume suffix; empty=auto-detect (_EVN_Vol first). '
                          'Use e.g. "_b8" for multi-bin runs.')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; filters volumes and provides '
                          'ref_defocus_um for stratified subset selection')
    inp.add_argument('--n-vols', type=int, default=None, metavar='N',
                     help='Number of volumes for training, stratified by defocus. '
                          'Default: use all volumes.')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', required=True,
                     help='Output directory (model checkpoints saved here)')

    tr = p.add_argument_group('training parameters')
    tr.add_argument('--model', '-m', default=None, metavar='PATH|NAME',
                    help='Starting model: path to a .sav checkpoint, or a '
                         'built-in name (unet-3d, unet-3d-10a, unet-3d-20a). '
                         'Default: topaz uses unet-3d.')
    tr.add_argument('--num-epochs', type=int, default=500,
                    help='Training epochs')
    tr.add_argument('--batch-size', type=int, default=10,
                    help='Mini-batch size')
    tr.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (adagrad optimizer)')
    tr.add_argument('--crop', type=int, default=96,
                    help='Training tile size in voxels')
    tr.add_argument('--n-train', type=int, default=1000,
                    dest='n_train',
                    help='Training patches sampled per volume')
    tr.add_argument('--n-test', type=int, default=200,
                    dest='n_test',
                    help='Validation patches sampled per volume')
    tr.add_argument('--save-interval', type=int, default=10,
                    help='Save a checkpoint every N epochs')
    tr.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Denoise volumes in patches of this size (voxels). '
                         'Default: topaz decides (usually full volume if it fits in memory).')
    tr.add_argument('--patch-padding', type=int, default=None, metavar='N',
                    help='Padding around each patch to reduce edge artefacts.')
    tr.add_argument('--bin', type=int, default=1, metavar='N',
                    help='Bin volumes by N before training (2 = half resolution). '
                         'Binned MRCs are written to training_data/evn_bN/ and '
                         'cached for re-runs.  Reduces RAM from loaded volumes '
                         'by N^3; allows higher --n-train.')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', type=int, default=-2,
                     help='GPU device: >=0 single GPU, -2 all GPUs, -1 CPU')
    ctl.add_argument('--num-workers', type=int, default=0,
                     help='DataLoader workers (default 0 — avoids forking '
                          'the pre-loaded dataset into each worker process)')
    ctl.add_argument('--topaz', dest='topaz_bin',
                     default='/opt/miniconda3/envs/topaz/bin/topaz',
                     help='Path to topaz executable')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print command without executing')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
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
        print('       Run run-aretomo3 cmd=2 with --split-sum 1 first.')
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
        selected_names = _stratified_sample(candidates, args.n_vols)
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

    # ── Check binary ──────────────────────────────────────────────────────
    if not args.dry_run:
        if shutil.which(args.topaz_bin) is None and not Path(args.topaz_bin).is_file():
            print(f'ERROR: topaz binary not found: {args.topaz_bin!r}')
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage training directories ────────────────────────────────────────
    bin_factor = getattr(args, 'bin', 1)
    if bin_factor > 1:
        evn_dir, odd_dir = _stage_binned_dirs(out_dir, train_pairs,
                                              bin_factor, dry_run=args.dry_run)
    else:
        evn_dir, odd_dir = _stage_training_dirs(out_dir, train_pairs,
                                                dry_run=args.dry_run)

    # ── Build command ─────────────────────────────────────────────────────
    model_prefix = str(out_dir / 'model')
    cmd = [
        args.topaz_bin, 'denoise3d',
        '-a', str(evn_dir),
        '-b', str(odd_dir),
        '--save-prefix',   model_prefix,
        '--num-epochs',    str(args.num_epochs),
        '--batch-size',    str(args.batch_size),
        '--lr',            str(args.lr),
        '-c',              str(args.crop),
        '--N-train',       str(args.n_train),
        '--N-test',        str(args.n_test),
        '--save-interval', str(args.save_interval),
        '--num-workers',   str(args.num_workers),
        '-d',              str(args.gpu),
    ]
    if args.model:
        cmd += ['-m', args.model]
    if args.patch_size is not None:
        cmd += ['-s', str(args.patch_size)]
    if args.patch_padding is not None:
        cmd += ['-p', str(args.patch_padding)]

    gpu_label = {-2: 'all GPUs', -1: 'CPU'}.get(args.gpu, f'GPU {args.gpu}')
    print(f'\n{prefix}Topaz command:')
    print(f'  $ {" ".join(cmd)}')
    print(f'\n{prefix}Training on {len(train_pairs)} volume pairs  |  {gpu_label}  |  {args.num_epochs} epochs')
    print(f'{prefix}Model checkpoints: {out_dir}/model_epoch*.sav')

    if args.dry_run:
        return

    # ── Run ───────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    result  = subprocess.run(cmd)
    elapsed = time.perf_counter() - t_start

    if result.returncode != 0:
        print(f'ERROR: topaz denoise3d exited with code {result.returncode}')
        sys.exit(result.returncode)

    print(f'\nTopaz training complete ({elapsed/60:.1f} min)')

    best_model = _find_latest_model(out_dir)
    if best_model:
        print(f'Latest checkpoint: {best_model}')
    print(f'\nTo denoise with this model:')
    print(f'  aretomo3-preprocess topaz-denoise3d \\')
    print(f'      --input {in_dir} --output {out_dir}/denoised \\')
    print(f'      --model {best_model or out_dir/"model_epochN.sav"} --gpu 0')

    update_section(
        section='topaz_train',
        values={
            'command':    ' '.join(cmd),
            'args':       args_to_dict(args),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
            'n_vols':     len(train_pairs),
            'bin_factor': bin_factor,
            'selected':   [p['ts_name'] for p in train_pairs],
            'output_dir': str(out_dir.resolve()),
            'model_prefix': model_prefix,
        },
        backup_dir=out_dir,
    )
