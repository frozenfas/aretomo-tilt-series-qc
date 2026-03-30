"""
simple-box-mask — create a simple box mask for each ts-*_Vol.mrc.

Generates a binary (0/1) MRC mask of the same dimensions as the input
tomogram.  Voxels are set to 1 inside a rectangular box defined by:

  XY plane : exclude <border> pixels from each edge
  Z  plane : include a slab of <thickness> pixels centred on the volume

No external binary required — uses mrcfile + numpy.

Typical usage
-------------
  aretomo3-preprocess simple-box-mask \\
      --input run002-cmd2-sart-thr80 \\
      --output run002-cmd2-sart-thr80/box_masks \\
      --border 25 \\
      --thickness 190

  # With QC report
  aretomo3-preprocess simple-box-mask \\
      --input run002-cmd2-sart-thr80 \\
      --output run002-cmd2-sart-thr80/box_masks \\
      --border 25 \\
      --thickness 190 \\
      --analyse

  # Dry run
  aretomo3-preprocess simple-box-mask \\
      --input run002-cmd2-sart-thr80 \\
      --output run002-cmd2-sart-thr80/box_masks \\
      --border 25 \\
      --thickness 190 \\
      --dry-run
"""

import re
import sys
import datetime
from pathlib import Path
import argparse

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.output_guard import check_output_dir
from aretomo3_preprocess.shared.volume_qc import orthoslices_with_mask_b64, make_ortho_html


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _make_mask(vol_path, border, thickness):
    """
    Create a binary uint8 mask for the given volume.

    Parameters
    ----------
    vol_path  : Path — input MRC volume
    border    : int  — pixels to zero at each XY edge
    thickness : int  — Z slab thickness in pixels (centred on volume)

    Returns
    -------
    (mask_array, voxel_size_x)  — numpy uint8 array (nz, ny, nx), float
    """
    import numpy as np
    import mrcfile

    with mrcfile.mmap(str(vol_path), mode='r', permissive=True) as mrc:
        nz, ny, nx = mrc.data.shape
        vox = float(mrc.voxel_size.x) or 1.0

    mask = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Z slab
    zc  = nz // 2
    hz  = max(1, thickness // 2)
    zs  = max(0,  zc - hz)
    ze  = min(nz, zc + hz)

    # XY interior
    xs = border
    xe = max(xs + 1, nx - border)
    ys = border
    ye = max(ys + 1, ny - border)

    mask[zs:ze, ys:ye, xs:xe] = 1
    return mask, vox, nz, ny, nx


def _write_mask(mask, vox, out_path):
    """Write a uint8 mask as an MRC file with the given voxel size."""
    import mrcfile
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(out_path), overwrite=True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = vox


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'simple-box-mask',
        help='Create a simple box mask (XY border + Z slab) for each tomogram',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    inp = p.add_argument_group('input')
    inp.add_argument('--input', '-i', required=True,
                     help='Directory containing ts-*_Vol.mrc files')
    inp.add_argument('--vol-suffix', default=None,
                     help='Extra suffix before _Vol.mrc (e.g. "_SART")')
    inp.add_argument('--select-ts', default=None, metavar='CSV',
                     help='ts-select.csv; only process selected TS')
    inp.add_argument('--include', nargs='+',
                     help='Process only these TS prefixes (wildcards supported)')
    inp.add_argument('--exclude', nargs='+',
                     help='Exclude these TS prefixes (wildcards supported)')

    out = p.add_argument_group('output')
    out.add_argument('--output', '-o', default='box_masks',
                     help='Output directory for mask MRCs (ts-xxx_mask.mrc)')

    msk = p.add_argument_group('mask parameters')
    msk.add_argument('--border', type=int, default=0, metavar='PX',
                     help='Pixels to exclude from each XY edge (default: 0)')
    msk.add_argument('--thickness', type=int, required=True, metavar='PX',
                     help='Z slab thickness in pixels to include around the centre')

    qc = p.add_argument_group('QC report')
    qc.add_argument('--analyse', action='store_true',
                    help='Generate an HTML report with orthogonal sections (XY, XZ, YZ) '
                         'of the tomogram with the mask overlaid in transparent blue')
    qc.add_argument('--analyse-output', default=None, metavar='HTML',
                    help='Path for QC report HTML '
                         '(default: <output>/simple_box_mask_qc.html)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove existing output directory before running')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print what would be done without writing any files')

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

    print(f'Tomograms to mask: {len(prefixes)}')
    print(f'Border            : {args.border} px')
    print(f'Z thickness       : {args.thickness} px')
    print(f'Output            : {out_dir}/')
    print(sep)
    for p in prefixes[:10]:
        print(f'  {p}')
    if len(prefixes) > 10:
        print(f'  ... ({len(prefixes) - 10} more)')
    print(sep)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    do_qc      = getattr(args, 'analyse', False)
    qc_entries = []

    ok, failed = [], []

    for i, prefix in enumerate(prefixes):
        print(f'\n[{i+1}/{len(prefixes)}] {prefix}', flush=True)

        tomo     = vol_map[prefix]
        mask_out = out_dir / f'{prefix}_mask.mrc'

        if args.dry_run:
            # Read dimensions without writing
            try:
                import mrcfile
                with mrcfile.mmap(str(tomo), mode='r', permissive=True) as mrc:
                    nz, ny, nx = mrc.data.shape
                    vox = float(mrc.voxel_size.x) or 1.0
                zc = nz // 2
                hz = max(1, args.thickness // 2)
                zs = max(0, zc - hz)
                ze = min(nz, zc + hz)
                print(f'  volume  : {nx} × {ny} × {nz} px  ({vox:.3f} Å/px)')
                print(f'  Z slab  : {zs}–{ze} (thickness {ze - zs} px)')
                print(f'  XY box  : x [{args.border}:{nx - args.border}]  '
                      f'y [{args.border}:{ny - args.border}]')
                print(f'  → {mask_out}  [dry-run: not written]')
            except Exception as exc:
                print(f'  WARNING: could not read volume ({exc})')
            ok.append(prefix)
            continue

        try:
            mask, vox, nz, ny, nx = _make_mask(tomo, args.border, args.thickness)
            _write_mask(mask, vox, mask_out)
            print(f'  volume  : {nx} × {ny} × {nz} px  ({vox:.3f} Å/px)', flush=True)
            print(f'  → {mask_out}', flush=True)
            ok.append(prefix)
        except Exception as exc:
            print(f'  ERROR: {exc}')
            failed.append(prefix)
            continue

        # QC
        if do_qc:
            img_b64 = orthoslices_with_mask_b64(tomo, mask_out)
            qc_entries.append({
                'ts_name':   prefix,
                'img_b64':   img_b64,
                'tomo_path': str(tomo),
                'mask_path': str(mask_out),
                'metadata': {
                    'border':    f'{args.border} px',
                    'thickness': f'{args.thickness} px',
                    'volume':    f'{nx} × {ny} × {nz}',
                    'voxel':     f'{vox:.3f} Å',
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
                     else out_dir / 'simple_box_mask_qc.html')
        make_ortho_html(
            entries  = qc_entries,
            out_path = html_path,
            title    = 'simple-box-mask QC',
            command  = ' '.join(sys.argv),
        )

    if args.dry_run:
        return

    update_section(
        section='simple_box_mask',
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
