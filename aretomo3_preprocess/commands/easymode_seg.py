"""
easymode-seg — cryoET reconstruction + segmentation via easymode.

Thin wrapper around the `easymode` CLI (https://github.com/mgflast/easymode):
  - optionally drives `easymode reconstruct` (WarpTools + AreTomo3) to go
    from raw frames + mdocs to reconstructed tomograms
  - then `easymode segment <feature...>` on the resulting (or a pre-existing)
    tomogram directory

Pretrained models (2026-07): ribosome/void/cytoplasm are validated at their
own trained pixel size (10/30/50 Å/px respectively); easymode auto-rescales
whatever pixel size the input .mrc header reports to match, so tomograms
don't need to be pre-binned per model. membrane is experimental and, per
easymode's own docs, "probably a bit worse" than membrain-seg (already
wrapped separately in this repo as membrain_seg.py) -- prefer that for
membranes.

easymode must be installed in the easymode conda environment at
/opt/miniconda3/envs/easymode/ (default), or be available on PATH.

`reconstruct` has no --output flag of its own -- it writes a warp_tiltseries/
project tree relative to the process's working directory, so this wrapper
runs it with cwd=--output and reads tomograms back from
<output>/warp_tiltseries/reconstruction/.

Typical usage
-------------
  # Reconstruct from raw frames, then segment ribosome + void + cytoplasm
  aretomo3-preprocess easymode-seg \\
      --frames frames/ --mdocs frames/ \\
      --feature ribosome void cytoplasm \\
      --output easymode_run --gpu 0

  # Segment only, from tomograms already reconstructed
  aretomo3-preprocess easymode-seg \\
      --input easymode_run/warp_tiltseries/reconstruction \\
      --feature ribosome void cytoplasm \\
      --output easymode_run --gpu 0

  # Only a subset of tilt series
  aretomo3-preprocess easymode-seg \\
      --input easymode_run/warp_tiltseries/reconstruction \\
      --feature void --output easymode_run \\
      --select-ts ts_selection.csv

  # Dry run to check commands
  aretomo3-preprocess easymode-seg \\
      --input easymode_run/warp_tiltseries/reconstruction \\
      --feature ribosome --output easymode_seg --dry-run
"""

import sys
import shutil
import datetime
import threading
import subprocess
from pathlib import Path
import argparse

from tqdm import tqdm

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir, check_disk_space
from aretomo3_preprocess.shared.discovery import find_volumes as _find_aretomo3_volumes

_EASYMODE_BIN = '/opt/miniconda3/envs/easymode/bin/easymode'

# .mrc files to never treat as "a tomogram to segment" in the generic
# (non-AreTomo3) fallback discovery below: easymode's own segmentation
# outputs (<name>__<feature>.mrc), AreTomo3 half-stacks/half-volumes (in
# case they end up alongside a flat file listing despite not matching the
# ts-*_Vol.mrc pattern _find_aretomo3_volumes looks for first), and common
# half-map/denoised naming patterns from other pipelines (WarpTools etc.).
_EXCLUDE_MARKERS = ('__', '_EVN', '_ODD', '_CTF', '_half1', '_half2', '_even', '_odd')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_easymode(easymode_dir=None):
    candidates = []
    if easymode_dir:
        candidates.append(str(Path(easymode_dir) / 'easymode'))
    candidates.append(_EASYMODE_BIN)
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which('easymode')


def _discover_tomograms(data_dir: Path) -> list:
    """
    Sorted [(name, path), ...] of tomograms to segment.

    Tries shared/discovery.py's find_volumes() first -- the same
    ts-*_Vol.mrc-aware logic membrain_seg.py/slabify.py already use, which
    correctly separates AreTomo3's actual 3D reconstructions from the
    ts-*.mrc/_EVN.mrc/_ODD.mrc 2D input stacks living in the same flat
    directory. Only if that finds nothing (e.g. a non-AreTomo3 source, like
    easymode's own WarpTools-style reconstruction output, which doesn't use
    the ts-* naming convention at all) falls back to a generic "every .mrc
    directly in this directory that isn't an obvious half-map/output" glob.
    """
    if not data_dir.is_dir():
        return []
    pairs = _find_aretomo3_volumes(data_dir)
    if pairs:
        return sorted(pairs)
    return sorted(
        (p.stem, p) for p in data_dir.glob('*.mrc')
        if not any(m in p.stem for m in _EXCLUDE_MARKERS)
    )


def _write_tomogram_list(paths: list, list_path: Path):
    list_path.write_text('\n'.join(str(p) for p in paths) + '\n')


def _run_with_progress(cmd: list, log_path: Path, cwd: Path, desc: str,
                       poll_count, total=None, interval: float = 5.0) -> int:
    """
    Run cmd with stdout/stderr redirected to a log file -- keeps the
    terminal to a single progress bar instead of raw TensorFlow/WarpTools
    output -- while a background thread polls poll_count() (a zero-arg
    callable returning the current "done" count, e.g. output files found on
    disk) to drive that bar. Never reads the subprocess's own stdout/stderr:
    the same reasoning as run_aretomo3.py's AreTomo3 invocation applies here
    (a pipe needs an active reader to keep draining it; a plain file
    redirect never blocks the child on a slow/stalled reader).
    """
    bar = tqdm(total=total, desc=desc, unit='file')
    stop_event = threading.Event()

    def _poll_loop():
        last_n = -1
        while not stop_event.is_set():
            try:
                n = poll_count()
                if n != last_n:
                    bar.n = min(n, bar.total) if bar.total is not None else n
                    bar.refresh()
                    last_n = n
            except Exception:
                pass
            stop_event.wait(interval)

    poll_thread = threading.Thread(target=_poll_loop, daemon=True)
    poll_thread.start()
    try:
        with open(log_path, 'wb') as log_fh:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(cwd))
            returncode = proc.wait()
    finally:
        stop_event.set()
        poll_thread.join(timeout=2)
        try:
            n = poll_count()
            bar.n = min(n, bar.total) if bar.total is not None else n
            bar.refresh()
        except Exception:
            pass
        bar.close()
    return returncode


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'easymode-seg',
        help='Reconstruct (optional) and segment tomograms with easymode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input (choose one)')
    inp.add_argument('--frames', default=None,
                     help='Directory of raw frames -- triggers `easymode reconstruct` first')
    inp.add_argument('--mdocs', default=None,
                     help='Directory of mdocs (required with --frames)')
    inp.add_argument('--input', '-i', default=None,
                     help='Directory of already-reconstructed tomograms '
                          '(skip reconstruction; segment these directly)')

    rec = p.add_argument_group('reconstruction (easymode reconstruct; only with --frames/--mdocs)')
    rec.add_argument('--apix', type=float, default=None,
                     help='Frame pixel size in Å (default: inferred from mdoc)')
    rec.add_argument('--axis', type=float, default=None,
                     help='Tilt axis orientation in degrees (default: inferred from mdoc -- '
                          'often wrong for Tomo5 collections)')
    rec.add_argument('--dose', type=float, default=None,
                     help='Dose per frame in e⁻/Å² (default: inferred from mdoc)')
    rec.add_argument('--extension', default=None,
                     help='Frame file extension (default: auto-detect)')
    rec.add_argument('--tomo-apix', type=float, default=10.0, metavar='ANGST',
                     help='Reconstructed tomogram pixel size in Å '
                          '(easymode networks are all trained at 10.0 Å/px)')
    rec.add_argument('--thickness', type=float, default=3000.0, metavar='ANGST',
                     help='Tomogram thickness in Å')
    rec.add_argument('--shape', default=None,
                     help='Frame shape, e.g. 4096x4096 (default: inferred)')
    rec.add_argument('--steps', default='11111111',
                     help='8-char string, one flag per reconstruction step '
                          '(1=run, 0=skip): motion+CTF, import tilt series, '
                          'create stacks, TS alignment, import alignments, '
                          'TS CTF, check handedness, reconstruct volumes')
    rec.add_argument('--no-halfmaps', action='store_true',
                     help="Don't generate half-maps (precludes most denoising methods)")
    rec.add_argument('--force-align', action='store_true',
                     help='Force AreTomo3 realignment even if alignment files already exist')

    sel = p.add_argument_group('TS selection')
    sel.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts_selection.csv; only process selected TS '
                          '(matched by tomogram filename stem)')
    sel.add_argument('--include', nargs='+', default=None,
                     help='Process only tomograms whose filename stem matches '
                          '(wildcards supported)')
    sel.add_argument('--exclude', nargs='+', default=None,
                     help='Exclude tomograms whose filename stem matches '
                          '(wildcards supported)')

    seg = p.add_argument_group('segmentation (easymode segment)')
    seg.add_argument('--feature', '-f', nargs='+', required=True,
                     help="Feature(s) to segment, e.g. 'ribosome void cytoplasm'. "
                          "Run 'easymode list' for the full, current model catalogue.")
    seg.add_argument('--tta', type=int, default=4,
                     help='Test-time augmentation factor, 1-16 (higher = better/slower)')
    seg.add_argument('--tile', default=None, metavar='ZxYxX',
                     help='Inference tile size (default 160x160x160; shrink if low on GPU memory)')
    seg.add_argument('--overlap', type=int, default=None,
                     help='Tile overlap in voxels (default 48)')
    seg.add_argument('--format', choices=['float32', 'uint16', 'int8'], default='int8',
                     help='Output segmentation dtype')
    seg.add_argument('--seg-apix', type=float, default=None, metavar='ANGST',
                     help="Override the input .mrc header's pixel size for rescaling "
                          "(0.0 disables rescaling entirely). Leave unset to trust the header.")
    seg.add_argument('--force-2d', action='store_true', help='Force 2D segmentation for all features')
    seg.add_argument('--force-3d', action='store_true', help='Force 3D segmentation for all features')
    seg.add_argument('--overwrite', action='store_true',
                     help='Overwrite existing segmentations in --output')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default='easymode_run',
                     help='Output directory. Reconstruction (if run) writes '
                          '<output>/warp_tiltseries/; segmentations go to <output>/segmented/')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--gpu', default='0',
                     help='Comma-separated GPU id(s)')
    ctl.add_argument('--easymode-dir', default=None,
                     help='Directory containing the easymode binary '
                          '(default: /opt/miniconda3/envs/easymode/bin/)')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing --output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print commands without running')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    sep = '─' * 70

    if not args.frames and not args.input:
        print('ERROR: give either --frames/--mdocs (reconstruct first) or --input '
              '(tomograms already exist)')
        sys.exit(1)
    if args.frames and not args.mdocs:
        print('ERROR: --frames requires --mdocs')
        sys.exit(1)

    out_dir = Path(args.output).resolve()
    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)

    warnings = check_disk_space(out_dir)
    for w in warnings:
        print(f'WARNING: {w}')
    if warnings and not args.dry_run:
        print()

    easymode_bin = _find_easymode(args.easymode_dir)
    if not easymode_bin:
        msg = (f'easymode not found.\n'
               f'  Expected at {_EASYMODE_BIN}\n'
               f'  Or specify: --easymode-dir /path/to/easymode/bin')
        if args.dry_run:
            print(f'WARNING: {msg} (dry-run: continuing)')
            easymode_bin = 'easymode'
        else:
            print(f'ERROR: {msg}')
            sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reconstruction (optional) ───────────────────────────────────────────
    if args.frames:
        frames_dir = Path(args.frames).resolve()
        mdocs_dir  = Path(args.mdocs).resolve()
        if not frames_dir.is_dir():
            print(f'ERROR: --frames {frames_dir} not found')
            sys.exit(1)
        if not mdocs_dir.is_dir():
            print(f'ERROR: --mdocs {mdocs_dir} not found')
            sys.exit(1)

        rec_cmd = [
            easymode_bin, 'reconstruct',
            '--frames', str(frames_dir),
            '--mdocs',  str(mdocs_dir),
            '--tomo_apix', str(args.tomo_apix),
            '--thickness', str(args.thickness),
            '--steps', args.steps,
        ]
        if args.apix is not None:
            rec_cmd += ['--apix', str(args.apix)]
        if args.axis is not None:
            rec_cmd += ['--axis', str(args.axis)]
        if args.dose is not None:
            rec_cmd += ['--dose', str(args.dose)]
        if args.extension is not None:
            rec_cmd += ['--extension', args.extension]
        if args.shape is not None:
            rec_cmd += ['--shape', args.shape]
        if args.no_halfmaps:
            rec_cmd += ['--no_halfmaps']
        if args.force_align:
            rec_cmd += ['--force_align']

        n_mdocs = len(list(mdocs_dir.glob('*.mdoc')))
        recon_dir = out_dir / 'warp_tiltseries' / 'reconstruction'

        print(f'Reconstruction: {n_mdocs} mdoc(s) found in {mdocs_dir}/')
        print(f'  $ {" ".join(rec_cmd)}')
        print(f'  (cwd={out_dir}, since `easymode reconstruct` has no --output of its own)')
        print(sep)

        if args.dry_run:
            print('  [dry-run: skipping execution]')
        else:
            log_path = out_dir / 'easymode_reconstruct.log'
            print(f'Log: {log_path}')
            returncode = _run_with_progress(
                rec_cmd, log_path, cwd=out_dir, desc='reconstruct',
                poll_count=lambda: len(_discover_tomograms(recon_dir)),
                total=n_mdocs or None,
            )
            if returncode != 0:
                print(f'ERROR: easymode reconstruct exited with code {returncode} '
                      f'(see {log_path})')
                sys.exit(returncode)
            print(f'Reconstruction done: {len(_discover_tomograms(recon_dir))} tomogram(s) '
                  f'in {recon_dir}/')

        data_dir = recon_dir
    else:
        data_dir = Path(args.input).resolve()
        if not data_dir.is_dir() and not args.dry_run:
            print(f'ERROR: --input {data_dir} not found')
            sys.exit(1)

    # ── Discover + filter tomograms for segmentation ───────────────────────
    all_pairs = _discover_tomograms(data_dir)   # [(name, path), ...]
    if not all_pairs and not args.dry_run:
        print(f'ERROR: no tomograms found in {data_dir}/')
        sys.exit(1)

    names = [n for n, _ in all_pairs]

    if args.include or args.exclude:
        import re as _re
        def _match_any(name, patterns):
            pats = patterns[0].split(',') if len(patterns) == 1 else patterns
            return any(_re.match(f'^{pat.replace("*", ".*")}$', name) for pat in pats)
        if args.include:
            keep = [n for n in names if _match_any(n, args.include)]
        else:
            keep = list(names)
        if args.exclude:
            keep = [n for n in keep if not _match_any(n, args.exclude)]
        n_excl = len(names) - len(keep)
        if n_excl:
            print(f'--include/--exclude: {n_excl} excluded, {len(keep)} remaining')
        names = keep

    selected_ts = resolve_selected_ts(args.select_ts)
    if selected_ts is not None:
        orig_n = len(names)
        names  = [n for n in names if n in selected_ts]
        n_excl = orig_n - len(names)
        if n_excl:
            print(f'TS selection: {n_excl} excluded, {len(names)} remaining')

    name_set = set(names)
    pairs = [(n, p) for n, p in all_pairs if n in name_set]

    if not pairs and not args.dry_run:
        print('ERROR: no tomograms to segment after filtering')
        sys.exit(1)

    print(f'\nTomograms to segment: {len(pairs)}')
    print(sep)
    for n, p in pairs[:10]:
        print(f'  {p.name}')
    if len(pairs) > 10:
        print(f'  ... ({len(pairs) - 10} more)')
    print(sep)

    # ── Segmentation ─────────────────────────────────────────────────────────
    features   = [f.lower() for f in args.feature]
    seg_out    = out_dir / 'segmented'
    data_arg   = str(data_dir)
    list_path  = None

    # If filtering actually removed something, point --data at an explicit
    # file list instead of the whole directory -- easymode's own --data
    # accepts a .txt file with one tomogram path per line for exactly this.
    if len(pairs) != len(all_pairs):
        list_path = out_dir / 'easymode_seg_tomograms.txt'
        if not args.dry_run:
            _write_tomogram_list([p for _, p in pairs], list_path)
        data_arg = str(list_path)

    seg_cmd = [
        easymode_bin, 'segment', *features,
        '--data',   data_arg,
        '--output', str(seg_out),
        '--tta',    str(args.tta),
        '--format', args.format,
        '--gpu',    args.gpu,
    ]
    if args.tile is not None:
        seg_cmd += ['--tile', args.tile]
    if args.overlap is not None:
        seg_cmd += ['--overlap', str(args.overlap)]
    if args.seg_apix is not None:
        seg_cmd += ['--apix', str(args.seg_apix)]
    if args.force_2d:
        seg_cmd += ['--2d']
    if args.force_3d:
        seg_cmd += ['--3d']
    if args.overwrite:
        seg_cmd += ['--overwrite']

    print(f'\nSegmentation: {len(pairs)} tomogram(s) × {len(features)} feature(s) '
          f'= {len(pairs) * len(features)} outputs')
    print(f'  $ {" ".join(seg_cmd)}')
    print(sep)

    if args.dry_run:
        print('  [dry-run: skipping execution]')
        return

    if list_path:
        print(f'Tomogram list: {list_path}')

    log_path = out_dir / 'easymode_segment.log'
    print(f'Log: {log_path}')

    total = len(pairs) * len(features)
    returncode = _run_with_progress(
        seg_cmd, log_path, cwd=out_dir, desc='segment',
        poll_count=lambda: len(list(seg_out.glob('*__*.mrc'))) if seg_out.is_dir() else 0,
        total=total,
    )

    n_done = len(list(seg_out.glob('*__*.mrc'))) if seg_out.is_dir() else 0
    print(f'\n{sep}')
    if returncode != 0:
        print(f'ERROR: easymode segment exited with code {returncode} (see {log_path})')
    print(f'Done: {n_done}/{total} segmentation output(s) in {seg_out}/')

    update_section(
        section='easymode_seg',
        values={
            'command':       ' '.join(sys.argv),
            'args':          args_to_dict(args),
            'timestamp':     datetime.datetime.now().isoformat(timespec='seconds'),
            'features':      features,
            'n_tomograms':   len(pairs),
            'n_outputs':     n_done,
            'data_dir':      str(data_dir),
            'output_dir':    str(seg_out),
            'reconstructed': bool(args.frames),
        },
        backup_dir=out_dir,
    )

    if returncode != 0:
        sys.exit(returncode)
