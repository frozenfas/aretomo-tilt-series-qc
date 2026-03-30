"""
membrain-seg — batch membrane segmentation with MemBrain-seg on AreTomo3 tomograms.

Runs `membrain segment` on each ts-*_Vol.mrc in an AreTomo3 output directory.
Requires a pre-trained model checkpoint (download from Zenodo; see
https://github.com/teamtomo/membrain-seg).

MemBrain-seg must be installed in the membrain-seg conda environment at
/opt/miniconda3/envs/membrain-seg/ (default), or be available on PATH.

Typical usage
-------------
  # Segment all tomograms
  aretomo3-preprocess membrain-seg \\
      --input run002-cmd2-sart-thr80 \\
      --checkpoint /path/to/MemBrain_seg_v10_alpha.ckpt \\
      --output membrain_seg

  # With connected-components filtering and probability maps
  aretomo3-preprocess membrain-seg \\
      --input run002-cmd2-sart-thr80 \\
      --checkpoint /path/to/MemBrain_seg_v10_alpha.ckpt \\
      --output membrain_seg \\
      --store-probabilities \\
      --connected-components 1000

  # Dry run to check commands
  aretomo3-preprocess membrain-seg \\
      --input run002-cmd2-sart-thr80 \\
      --checkpoint /path/to/MemBrain_seg_v10_alpha.ckpt \\
      --output membrain_seg \\
      --dry-run
"""

import re
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir
from aretomo3_preprocess.shared.volume_qc import (
    central_slab_projection, projection_to_b64png,
    slab_with_mask_b64, make_comparison_html,
)

_MEMBRAIN_BIN = '/opt/miniconda3/envs/membrain-seg/bin/membrain'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_membrain(membrain_dir=None):
    candidates = []
    if membrain_dir:
        candidates.append(str(Path(membrain_dir) / 'membrain'))
    candidates.append(_MEMBRAIN_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('membrain')


def _find_volumes(in_dir, vol_suffix=None):
    """Return sorted list of (prefix, vol_path) tuples."""
    if vol_suffix:
        vol_glob = f'ts-*{vol_suffix}_Vol.mrc'
    else:
        vol_glob = 'ts-*_Vol.mrc'

    vols = [v for v in sorted(in_dir.glob(vol_glob))
            if '_EVN' not in v.name and '_ODD' not in v.name]

    if not vols and not vol_suffix:
        # Fallback: ts-*.mrc (older AreTomo3 output without _Vol suffix)
        vols = [v for v in sorted(in_dir.glob('ts-*.mrc'))
                if not any(t in v.name for t in ('_EVN', '_ODD', '_CTF'))]

    def _prefix(v):
        name = v.stem
        for tag in ('_Vol', vol_suffix or ''):
            if tag and name.endswith(tag):
                name = name[:-len(tag)]
        return name

    return [(_prefix(v), v) for v in vols]


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


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'membrain-seg',
        help='Batch membrane segmentation with MemBrain-seg',
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

    seg = p.add_argument_group('segmentation (membrain segment)')
    seg.add_argument('--checkpoint', required=True,
                     help='Path to pre-trained MemBrain-seg model checkpoint (.ckpt)')
    seg.add_argument('--output', '-o', default='membrain_seg',
                     help='Output directory (per-TS subdirectories will be created)')
    seg.add_argument('--in-pixel-size', type=float, default=None, metavar='ANGST',
                     help='Tomogram pixel size in Å (auto-read from project.json if omitted)')
    seg.add_argument('--out-pixel-size', type=float, default=10.0, metavar='ANGST',
                     help='Output segmentation pixel size in Å (MemBrain default: 10 Å)')
    seg.add_argument('--segmentation-threshold', type=float, default=None,
                     help='Membrane score threshold (default: 0.0 — keep all voxels above 0)')
    seg.add_argument('--sliding-window-size', type=int, default=None,
                     help='Sliding window size for inference (default 160; smaller = less GPU)')
    seg.add_argument('--rescale-patches', action='store_true',
                     help='Rescale patches on-the-fly during inference')
    seg.add_argument('--test-time-augmentation', action='store_true',
                     help='Use 8-fold test-time augmentation (improves quality, slower)')
    seg.add_argument('--store-probabilities', action='store_true',
                     help='Also write probability map MRC alongside segmentation')
    seg.add_argument('--connected-components', type=int, default=None, metavar='THRESHOLD',
                     help='Remove connected components smaller than this voxel count')

    ctl = p.add_argument_group('run control')
    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report with central-slab tomogram and '
                         'segmentation overlay side by side')
    qc.add_argument('--analyse-thickness', type=float, default=300.0, metavar='ANGST',
                    help='Slab thickness in Å for QC projections (default: 300 Å)')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/membrain_seg_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--membrain-dir', default=None,
                     help='Directory containing the membrain binary '
                          '(default: /opt/miniconda3/envs/membrain-seg/bin/)')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without running')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    sep     = '─' * 70

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f'ERROR: --checkpoint {ckpt} not found')
        sys.exit(1)

    # Locate membrain binary
    membrain_bin = _find_membrain(args.membrain_dir)
    if not membrain_bin:
        msg = (f'membrain not found.\n'
               f'  Expected at {_MEMBRAIN_BIN}\n'
               f'  Or specify: --membrain-dir /path/to/membrain-seg/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            membrain_bin = 'membrain'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    # Find volumes
    pairs = _find_volumes(in_dir, args.vol_suffix)
    if not pairs:
        print(f'ERROR: no tomogram volumes found in {in_dir}/')
        sys.exit(1)

    # Resolve pixel size: --in-pixel-size > MRC header > warn
    angpix = args.in_pixel_size
    if angpix is None:
        try:
            import mrcfile
            with mrcfile.mmap(str(pairs[0][1]), mode='r', permissive=True) as mrc:
                px = float(mrc.voxel_size.x)
            if px > 0:
                angpix = px
                print(f'Pixel size: {angpix} Å  (from MRC header of {pairs[0][1].name})')
        except Exception:
            pass
    if angpix is None:
        print('WARNING: --in-pixel-size not given and could not be read from MRC header; '
              'MemBrain will use its default (10 Å).')
    elif args.in_pixel_size is not None:
        print(f'Pixel size: {angpix} Å  (from --in-pixel-size)')

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

    print(f'Tomograms to segment: {len(prefixes)}')
    print(sep)
    for p in prefixes[:10]:
        print(f'  {p}')
    if len(prefixes) > 10:
        print(f'  ... ({len(prefixes) - 10} more)')
    print(sep)

    do_qc    = getattr(args, 'analyse', False)
    qc_thick = getattr(args, 'analyse_thickness', 300.0)
    qc_entries = []

    ok, failed = [], []

    for i, prefix in enumerate(prefixes):
        print(f'\n[{i+1}/{len(prefixes)}] {prefix}')

        tomo     = vol_map[prefix]
        ts_out   = out_dir / prefix

        cmd = [
            membrain_bin, 'segment',
            '--tomogram-path', str(tomo),
            '--ckpt-path',     str(ckpt),
            '--out-folder',    str(ts_out),
        ]
        if angpix is not None:
            cmd += ['--in-pixel-size', str(angpix)]
        cmd += ['--out-pixel-size', str(args.out_pixel_size)]
        if args.segmentation_threshold is not None:
            cmd += ['--segmentation-threshold', str(args.segmentation_threshold)]
        if args.sliding_window_size is not None:
            cmd += ['--sliding-window-size', str(args.sliding_window_size)]
        if args.rescale_patches:
            cmd += ['--rescale-patches']
        if args.test_time_augmentation:
            cmd += ['--test-time-augmentation']
        if args.store_probabilities:
            cmd += ['--store-probabilities']
        if args.connected_components is not None:
            cmd += ['--store-connected-components',
                    '--connected-component-thres', str(args.connected_components)]

        _print_cmd(cmd)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
            ok.append(prefix)
            continue

        ts_out.mkdir(parents=True, exist_ok=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'  ERROR: membrain segment exited with code {ret.returncode}')
            failed.append(prefix)
            continue

        ok.append(prefix)

        # Report output files
        seg_files = sorted(ts_out.glob('*_segmentation.mrc'))
        for f in seg_files:
            print(f'  → {f}')

        # QC
        if do_qc:
            before_b64 = after_b64 = None
            proj = central_slab_projection(tomo, qc_thick)
            if proj:
                before_b64 = projection_to_b64png(proj['img'])
            if seg_files:
                after_b64 = slab_with_mask_b64(tomo, seg_files[0], qc_thick)
            qc_entries.append({
                'ts_name':    prefix,
                'before_b64': before_b64,
                'after_b64':  after_b64,
                'before_path': str(tomo),
                'after_path':  str(seg_files[0]) if seg_files else '',
                'metadata': {
                    'pixel size': f'{angpix} Å' if angpix else 'unknown',
                    'out pixel':  f'{args.out_pixel_size} Å',
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
                     else out_dir / 'membrain_seg_qc.html')
        make_comparison_html(
            entries      = qc_entries,
            out_path     = html_path,
            title        = 'membrain-seg QC',
            command      = ' '.join(sys.argv),
            before_label = 'Tomogram',
            after_label  = 'Segmentation overlay',
            slab_angst   = qc_thick,
        )

    if args.dry_run:
        return

    update_section(
        section='membrain_seg',
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
