"""
trim-ts subcommand — for each tilt series write per-variant .tlt / .xtilt
files and newst.com scripts in the analysis directory, with dark and/or
misaligned frames excluded.

Two variants are produced per tilt series:

    nodark  —  dark frames removed
    clean   —  dark + overlap-flagged frames removed

Files written to  <output>/ts-xxx_Imod/:

    ts-xxx_st.xf    (symlink)           → original full-length transform file
    ts-xxx_nodark.xtilt / .tlt          trimmed support files for nodark ali
    ts-xxx_clean.xtilt  / .tlt          trimmed support files for clean ali
    ts-xxx_nodark_order_list.csv        acquisition metadata for nodark frames
    ts-xxx_clean_order_list.csv         acquisition metadata for clean frames
    newst_nodark.com                    newstack script (dark removed, bin 8)
    newst_clean.com                     newstack script (dark+misaligned, bin 8)
    ts-xxx.mrc  (symlink)               → raw tilt-sorted stack

Requires:
    aretomo3-preprocess analyse to have been run first  (reads alignment_data.json)
    Source _Imod directories present in <input>/ts-xxx_Imod/
"""

import sys
import json
import math
import subprocess
from pathlib import Path

from aretomo3_preprocess.shared.project_state import get_input_stacks, resolve_selected_ts

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):
        return it


def _calc_output_size(w, h, rot_deg, bin_factor=1):
    """
    Output image dimensions (pixels, pre-bin) needed to fully contain a W×H
    image rotated in-plane by rot_deg degrees, without clipping.

    Rounded up to the next multiple of bin_factor so that after BinByFactor N
    the output is an exact integer number of pixels in both dimensions.
    """
    a  = math.radians(rot_deg)
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    out_x = math.ceil(w * ca + h * sa)
    out_y = math.ceil(w * sa + h * ca)
    m = max(2, bin_factor)          # at minimum round to even
    out_x = math.ceil(out_x / m) * m
    out_y = math.ceil(out_y / m) * m
    return out_x, out_y


# ─────────────────────────────────────────────────────────────────────────────
# Order-list helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_order_list(csv_path):
    """
    Parse ts-xxx_order_list.csv.
    Returns [(img_num_1indexed, tilt_angle), ...] in acquisition order.
    """
    rows = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or 'ImageNumber' in line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                rows.append((int(parts[0].strip()), float(parts[1].strip())))
    return rows


def tilt_sorted_sections(order_rows):
    """
    Sort by tilt angle (ascending) — matches the section order in ts-xxx.mrc.
    Returns [(section_0indexed, tilt_angle, acq_img_num), ...].
    """
    sorted_rows = sorted(order_rows, key=lambda r: r[1])
    return [(i, tilt, img_num) for i, (img_num, tilt) in enumerate(sorted_rows)]


def find_sections_by_tilt(target_tilts, sorted_secs, tol=2.0):
    """
    Match each target tilt to the closest section in the sorted stack.
    Returns (matched_set_of_0indexed_sections, list_of_unmatched_tilts).
    """
    matched, unmatched = set(), []
    for target in target_tilts:
        best_sec, best_diff = None, float('inf')
        for sec_idx, tilt, _ in sorted_secs:
            diff = abs(tilt - target)
            if diff < best_diff:
                best_diff = diff
                best_sec = sec_idx
        if best_diff <= tol and best_sec is not None:
            matched.add(best_sec)
        else:
            unmatched.append(target)
    return matched, unmatched


# ─────────────────────────────────────────────────────────────────────────────
# Source-file readers
# ─────────────────────────────────────────────────────────────────────────────

def read_xtilt(path):
    """Read a .xtilt file; return list of stripped strings (one per section)."""
    with open(path) as fh:
        return [line.rstrip() for line in fh if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Trimmed-file writers
# ─────────────────────────────────────────────────────────────────────────────

def write_trimmed(path, all_lines, keep_indices):
    """Write subset of *all_lines* selected by *keep_indices* (0-indexed)."""
    kept = [all_lines[i] for i in keep_indices if i < len(all_lines)]
    Path(path).write_text('\n'.join(kept) + '\n')


def write_tlt(path, sorted_secs, keep_set):
    """Write tilt angles for sections in *keep_set*, in tilt-sorted order."""
    lines = [f'{tilt:8.2f}' for sec_idx, tilt, _ in sorted_secs
             if sec_idx in keep_set]
    Path(path).write_text('\n'.join(lines) + '\n')


def write_order_list(path, sorted_secs, keep_set):
    """Write trimmed order_list.csv (acquisition metadata) for kept sections."""
    lines = ['ImageNumber, TiltAngle']
    new_num = 1
    for sec_idx, tilt, acq_img_num in sorted_secs:
        if sec_idx in keep_set:
            lines.append(f'{new_num:4d},{tilt:8.2f}')
            new_num += 1
    Path(path).write_text('\n'.join(lines) + '\n')


def write_newst(path, ts_name, xf_name, output_ali, bin_factor, keep_indices,
                out_x=None, out_y=None):
    """
    Write a newst.com that uses the trimmed xf file.

    SectionsToRead (0-indexed) tells newstack which sections to pull from the
    full raw .mrc.  IMOD newstack uses section-number indexing for the xf
    (verified experimentally), so the full unmodified xf is safe to use.

    out_x / out_y: output image size in pixels (pre-bin).  When the xf
    includes a large in-plane rotation, specifying a larger output size avoids
    clipping the image content at the top/bottom.  Calculated from the ROT
    angle by _calc_output_size() if not supplied.

    SizeToOutputInXandY is specified in post-bin (output) pixels, so out_x
    and out_y are divided by bin_factor before writing.  _calc_output_size
    rounds them to a multiple of bin_factor so the division is always exact.
    """
    sections_str = ','.join(str(i) for i in keep_indices)
    if out_x and out_y:
        size_line = (f'SizeToOutputInXandY\t'
                     f'{out_x // bin_factor},{out_y // bin_factor}\n')
    else:
        size_line = ''
    content = (
        f'$newstack -StandardInput\n'
        f'InputFile\t{ts_name}.mrc\n'
        f'OutputFile\t{output_ali}\n'
        f'TransformFile\t{xf_name}\n'
        f'SectionsToRead\t{sections_str}\n'
        f'{size_line}'
        f'TaperAtFill     1,0\n'
        f'AdjustOrigin\n'
        f'OffsetsInXandY  0.0,0.0\n'
        f'ImagesAreBinned 1.0\n'
        f'BinByFactor     {bin_factor}\n'
        f'$if (-e ./savework) ./savework\n'
    )
    Path(path).write_text(content)


def _run_submfg(work_dir, com_file, ts_name, pwrite=print):
    """Run submfg <com_file> from work_dir synchronously. Print result."""
    pwrite(f'    submfg {com_file} ...')
    try:
        result = subprocess.run(
            ['submfg', com_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pwrite(f'    submfg {com_file}: FAILED — submfg not found in PATH')
        return
    if result.returncode == 0:
        pwrite(f'    submfg {com_file}: OK')
    else:
        pwrite(f'    submfg {com_file}: FAILED (exit {result.returncode})')
        for line in (result.stdout + result.stderr).splitlines()[-5:]:
            if line.strip():
                pwrite(f'      {line}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI integration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'trim-ts',
        help='Trim IMOD support files for dark/misaligned frame removal',
        formatter_class=__import__('argparse').ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',     '-i', required=True,
                   help='Directory containing ts-xxx_Imod subdirectories')
    p.add_argument('--output',    '-o', default='run002_analysis',
                   help='Analysis output directory (must contain alignment_data.json)')
    p.add_argument('--threshold', '-t', type=float, default=80.0,
                   help='Overlap threshold used in the analyse step')
    p.add_argument('--bin',       '-b', type=int, default=8,
                   help='BinByFactor for newstack')
    p.add_argument('--select-ts', default=None, metavar='CSV',
                   help='Path to ts_selection.csv from select-ts; only the '
                        'selected TS are processed.')
    p.add_argument('--run-submfg', choices=['none', 'nodark', 'clean', 'both'],
                   default='none',
                   help='Run submfg on the generated .com files after writing them. '
                        '"nodark" runs newst_nodark.com, "clean" runs newst_clean.com, '
                        '"both" runs both (default: none).')
    p.set_defaults(func=run)
    return p


def run(args):
    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()

    json_path = out_dir / 'alignment_data.json'
    if not json_path.exists():
        print(f'ERROR: {json_path} not found — run "aretomo3-preprocess analyse" first.')
        return

    with open(json_path) as fh:
        all_parsed = json.load(fh)

    # ── TS selection filter ───────────────────────────────────────────────────
    selected_ts = resolve_selected_ts(getattr(args, 'select_ts', None))
    if selected_ts is not None:
        n_before   = len(all_parsed)
        all_parsed = {k: v for k, v in all_parsed.items() if k in selected_ts}
        print(f'TS selection: {len(all_parsed)}/{n_before} TS kept')

    # ── MRC stacks from project.json ──────────────────────────────────────────
    stacks = get_input_stacks()
    if stacks is None:
        print('ERROR: input_stacks not found in project.json.')
        print('       Register the cmd=0 MRC stacks:')
        print('         aretomo3-preprocess enrich --mrc-data <run001/>')
        sys.exit(1)

    print(f'Trimming IMOD support files into : {out_dir}')
    print(f'Source _Imod directories         : {in_dir}')
    print(f'Bin factor                       : {args.bin}')
    print(f'Overlap threshold                : {args.threshold}%\n')

    # Wipe link folders now so they are never stale even if the run is
    # interrupted before reaching the link-creation section at the end.
    import shutil as _shutil
    for _variant in ('clean', 'nodark'):
        _link_dir = out_dir / f'{_variant}_ts'
        if _link_dir.exists():
            _shutil.rmtree(_link_dir)
            print(f'Cleared {_link_dir.name}/')
    print()

    sep = '─' * 68
    n_done = n_skip = 0
    processed_ts = []   # TS names successfully processed (for link folders)

    _pbar = _tqdm(list(all_parsed.items()), desc='trim-ts', unit='TS', ncols=80)
    _pwrite = getattr(_pbar, 'write', print)

    for ts_name, data in _pbar:
        imod_src  = in_dir / f'{ts_name}_Imod'
        order_csv = imod_src / f'{ts_name}_order_list.csv'
        xf_src    = imod_src / f'{ts_name}_st.xf'
        xtilt_src = imod_src / f'{ts_name}_st.xtilt'

        missing = [p for p in (imod_src, order_csv, xf_src) if not p.exists()]
        if missing:
            _pwrite(f'SKIP {ts_name}: missing {[p.name for p in missing]}')
            n_skip += 1
            continue

        # Build tilt-sorted section index
        order_rows  = load_order_list(order_csv)
        sorted_secs = tilt_sorted_sections(order_rows)
        all_sec_idx = {s[0] for s in sorted_secs}

        # Identify sections to exclude
        dark_tilts    = [df['tilt'] for df in data.get('dark_frames', [])]
        flagged_tilts = [f['tilt']  for f  in data.get('frames', [])
                         if f.get('is_flagged')]

        dark_secs,    dark_miss    = find_sections_by_tilt(dark_tilts,    sorted_secs, tol=1.0)
        flagged_secs, flagged_miss = find_sections_by_tilt(flagged_tilts, sorted_secs, tol=2.0)

        for miss, label in ((dark_miss, 'dark'), (flagged_miss, 'flagged')):
            if miss:
                _pwrite(f'  WARNING {ts_name}: {len(miss)} {label} tilt(s) unmatched: '
                        f'{[f"{t:.1f}" for t in miss]}')

        nodark_keep = sorted(all_sec_idx - dark_secs)
        clean_keep  = sorted(all_sec_idx - dark_secs - flagged_secs)

        xtilt_lines = read_xtilt(xtilt_src) if xtilt_src.exists() else []

        # Output size: expand to avoid clipping the in-plane rotation
        rot = data['frames'][0]['rot'] if data.get('frames') else 0.0
        W   = data.get('width',  0)
        H   = data.get('height', 0)
        out_x, out_y = _calc_output_size(W, H, rot, args.bin) if (W and H and rot) else (None, None)

        ts_out = out_dir / f'{ts_name}_Imod'
        ts_out.mkdir(exist_ok=True)

        # Symlink raw stack and full .xf
        mrc_link = ts_out / f'{ts_name}.mrc'
        if not mrc_link.exists():
            mrc_info = stacks.get(ts_name)
            if mrc_info:
                mrc_src = Path(mrc_info['path'])
                mrc_link.symlink_to(mrc_src)
                if not mrc_src.exists():
                    _pwrite(f'  WARNING {ts_name}: source .mrc not found at {mrc_src}')
            else:
                _pwrite(f'  WARNING {ts_name}: not in project.json input_stacks'
                        f' — .mrc symlink skipped')

        xf_link = ts_out / f'{ts_name}_st.xf'
        if not xf_link.exists():
            xf_link.symlink_to(xf_src)

        # Write nodark variant
        write_tlt(ts_out / f'{ts_name}_nodark.tlt', sorted_secs, set(nodark_keep))
        write_order_list(ts_out / f'{ts_name}_nodark_order_list.csv',
                         sorted_secs, set(nodark_keep))
        if xtilt_lines:
            write_trimmed(ts_out / f'{ts_name}_nodark.xtilt', xtilt_lines, nodark_keep)

        write_newst(
            ts_out / 'newst_nodark.com',
            ts_name,
            xf_name      = f'{ts_name}_st.xf',
            output_ali   = f'{ts_name}_nodark.ali',
            bin_factor   = args.bin,
            keep_indices = nodark_keep,
            out_x        = out_x,
            out_y        = out_y,
        )

        # Write clean variant
        write_tlt(ts_out / f'{ts_name}_clean.tlt', sorted_secs, set(clean_keep))
        write_order_list(ts_out / f'{ts_name}_clean_order_list.csv',
                         sorted_secs, set(clean_keep))
        if xtilt_lines:
            write_trimmed(ts_out / f'{ts_name}_clean.xtilt', xtilt_lines, clean_keep)

        write_newst(
            ts_out / 'newst_clean.com',
            ts_name,
            xf_name      = f'{ts_name}_st.xf',
            output_ali   = f'{ts_name}_clean.ali',
            bin_factor   = args.bin,
            keep_indices = clean_keep,
            out_x        = out_x,
            out_y        = out_y,
        )

        n_total  = len(all_sec_idx)
        n_dark   = len(dark_secs)
        n_extra  = len(flagged_secs - dark_secs)
        _pwrite(f'{sep}')
        _pwrite(f'  {ts_name}')
        _pwrite(f'    Total sections         : {n_total}')
        _pwrite(f'    Dark excluded          : {n_dark}')
        _pwrite(f'    Misaligned (extra)     : {n_extra}')
        _pwrite(f'    nodark keeps           : {len(nodark_keep)}  sections')
        _pwrite(f'    clean  keeps           : {len(clean_keep)}  sections')

        if args.run_submfg in ('nodark', 'both'):
            _run_submfg(ts_out, 'newst_nodark.com', ts_name, pwrite=_pwrite)
        if args.run_submfg in ('clean', 'both'):
            _run_submfg(ts_out, 'newst_clean.com', ts_name, pwrite=_pwrite)

        processed_ts.append(ts_name)
        n_done += 1

    # ── Create clean_ts/ and nodark_ts/ link folders ──────────────────────────
    for variant in ('clean', 'nodark'):
        link_dir = out_dir / f'{variant}_ts'
        # Remove and recreate on every run so stale links from a previous
        # run (different TS set or different filter) don't accumulate.
        if link_dir.exists():
            _shutil.rmtree(link_dir)
        link_dir.mkdir()
        n_tlt = n_ali = 0
        for ts_name in processed_ts:
            ts_imod = out_dir / f'{ts_name}_Imod'
            rel_imod = Path('..') / ts_imod.name
            # .tlt — always present after trim-ts
            tlt_target = ts_imod / f'{ts_name}_{variant}.tlt'
            if tlt_target.exists():
                (link_dir / f'{ts_name}.tlt').symlink_to(rel_imod / tlt_target.name)
                n_tlt += 1
            # .ali — only link if the file actually exists (created by submfg)
            ali_target = ts_imod / f'{ts_name}_{variant}.ali'
            if ali_target.exists():
                (link_dir / f'{ts_name}.ali').symlink_to(rel_imod / ali_target.name)
                n_ali += 1
        print(f'{variant}_ts/  {n_tlt} .tlt links,  {n_ali} .ali links  → {link_dir}')

    print(sep)
    print(f'\nDone: {n_done} TS processed, {n_skip} skipped')
    print(f'\nFiles written per TS into {out_dir}/ts-xxx_Imod/:')
    print(f'  ts-xxx_st.xf          (symlink — full stack, unmodified)')
    print(f'  ts-xxx_nodark.xtilt / .tlt / _order_list.csv')
    print(f'  ts-xxx_clean.xtilt  / .tlt / _order_list.csv')
    print(f'  newst_nodark.com  /  newst_clean.com')
    if args.run_submfg == 'none':
        print(f'\nTo run (from inside each ts-xxx_Imod directory):')
        print(f'  submfg newst_nodark.com')
        print(f'  submfg newst_clean.com')
        print(f'  (or rerun with --run-submfg nodark/clean/both)')
    print(f'\nLink folders for easy loading:')
    print(f'  {out_dir}/clean_ts/   — ts-xxx.ali + ts-xxx.tlt (clean variant)')
    print(f'  {out_dir}/nodark_ts/  — ts-xxx.ali + ts-xxx.tlt (nodark variant)')
