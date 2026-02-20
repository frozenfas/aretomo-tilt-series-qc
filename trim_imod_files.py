#!/usr/bin/env python3
"""
trim_imod_files.py  —  For each tilt series create per-variant trimmed copies
                        of the IMOD support files (.xf, .xtilt, .tlt) and write
                        matching newst.com files, in the analysis output directory.

Two variants are produced per tilt series:

    nodark  —  dark frames removed
    clean   —  dark + overlap-flagged frames removed

Files written to  <output>/ts-xxx_Imod/:

    ts-xxx_nodark.xf / .xtilt / .tlt   transform + tilt files for nodark variant
    ts-xxx_clean.xf  / .xtilt / .tlt   transform + tilt files for clean variant
    ts-xxx_nodark_order_list.csv        acquisition metadata for nodark frames
    ts-xxx_clean_order_list.csv         acquisition metadata for clean frames
    newst_nodark.com                    newstack script (dark removed, bin 8)
    newst_clean.com                     newstack script (dark+misaligned, bin 8)
    ts-xxx.mrc  (symlink)               → raw tilt-sorted stack

Each newst.com uses the matching trimmed .xf so that newstack receives exactly
one transform per input section it reads — no SectionsToRead needed.

Requires:
    parse_aln.py to have been run first  (reads <output>/alignment_data.json)
    Source _Imod directories present in <input>/ts-xxx_Imod/

Usage:
    python trim_imod_files.py \\
        --input    run002 \\
        --output   run002_analysis \\
        --mrcdir   /mnt/McQueen-002/parry/bi38262-21-akinetes/relion/run002 \\
        --threshold 80 \\
        --bin 8
"""

import json
import argparse
from pathlib import Path


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

def read_xf(path):
    """Read all lines of a .xf file; return as list of stripped strings."""
    with open(path) as fh:
        return [line.rstrip() for line in fh if line.strip()]


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


def write_newst(path, ts_name, xf_name, output_ali, bin_factor, keep_indices):
    """
    Write a newst.com that uses the trimmed xf file.

    SectionsToRead (0-indexed) tells newstack which sections to pull from the
    full raw .mrc.  The trimmed xf has exactly one transform per kept section
    in the same order, so the two are consistent.
    """
    sections_str = ','.join(str(i) for i in keep_indices)
    content = (
        f'$newstack -StandardInput\n'
        f'InputFile\t{ts_name}.mrc\n'
        f'OutputFile\t{output_ali}\n'
        f'TransformFile\t{xf_name}\n'
        f'SectionsToRead\t{sections_str}\n'
        f'TaperAtFill     1,0\n'
        f'AdjustOrigin\n'
        f'OffsetsInXandY  0.0,0.0\n'
        f'ImagesAreBinned 1.0\n'
        f'BinByFactor     {bin_factor}\n'
        f'$if (-e ./savework) ./savework\n'
    )
    Path(path).write_text(content)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Trim IMOD support files for dark/misaligned frame removal.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--input',     '-i', default='run002',
                    help='Directory containing ts-xxx_Imod subdirectories')
    ap.add_argument('--output',    '-o', default='run002_analysis',
                    help='Analysis output directory (must contain alignment_data.json)')
    ap.add_argument('--mrcdir',    '-m',
                    default='/mnt/McQueen-002/parry/bi38262-21-akinetes/relion/run002',
                    help='Directory containing the ts-xxx.mrc raw stacks')
    ap.add_argument('--threshold', '-t', type=float, default=80.0,
                    help='Overlap threshold used in parse_aln.py')
    ap.add_argument('--bin',       '-b', type=int, default=8,
                    help='BinByFactor for newstack')
    args = ap.parse_args()

    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    mrc_dir = Path(args.mrcdir).resolve()

    json_path = out_dir / 'alignment_data.json'
    if not json_path.exists():
        print(f'ERROR: {json_path} not found — run parse_aln.py first.')
        return

    with open(json_path) as fh:
        all_parsed = json.load(fh)

    print(f'Trimming IMOD support files into : {out_dir}')
    print(f'Source _Imod directories         : {in_dir}')
    print(f'Raw stack symlinks → mrcdir      : {mrc_dir}')
    print(f'Bin factor                       : {args.bin}')
    print(f'Overlap threshold                : {args.threshold}%\n')

    sep = '─' * 68
    n_done = n_skip = 0

    for ts_name, data in all_parsed.items():
        imod_src  = in_dir / f'{ts_name}_Imod'
        order_csv = imod_src / f'{ts_name}_order_list.csv'
        xf_src    = imod_src / f'{ts_name}_st.xf'
        xtilt_src = imod_src / f'{ts_name}_st.xtilt'

        missing = [p for p in (imod_src, order_csv, xf_src) if not p.exists()]
        if missing:
            print(f'SKIP {ts_name}: missing {[p.name for p in missing]}')
            n_skip += 1
            continue

        # ── Build tilt-sorted section index ──────────────────────────────────
        order_rows  = load_order_list(order_csv)
        sorted_secs = tilt_sorted_sections(order_rows)
        all_sec_idx = {s[0] for s in sorted_secs}

        # ── Identify sections to exclude ──────────────────────────────────────
        dark_tilts    = [df['tilt'] for df in data.get('dark_frames', [])]
        flagged_tilts = [f['tilt']  for f  in data.get('frames', [])
                         if f.get('is_flagged')]

        dark_secs,    dark_miss    = find_sections_by_tilt(dark_tilts,    sorted_secs, tol=1.0)
        flagged_secs, flagged_miss = find_sections_by_tilt(flagged_tilts, sorted_secs, tol=2.0)

        for miss, label in ((dark_miss, 'dark'), (flagged_miss, 'flagged')):
            if miss:
                print(f'  WARNING {ts_name}: {len(miss)} {label} tilt(s) unmatched: '
                      f'{[f"{t:.1f}" for t in miss]}')

        nodark_keep = sorted(all_sec_idx - dark_secs)
        clean_keep  = sorted(all_sec_idx - dark_secs - flagged_secs)

        # ── Read source transform files ───────────────────────────────────────
        xf_lines    = read_xf(xf_src)
        xtilt_lines = read_xtilt(xtilt_src) if xtilt_src.exists() else []

        # ── Create output directory ───────────────────────────────────────────
        ts_out = out_dir / f'{ts_name}_Imod'
        ts_out.mkdir(exist_ok=True)

        # ── Symlink raw stack ─────────────────────────────────────────────────
        mrc_link = ts_out / f'{ts_name}.mrc'
        if not mrc_link.exists():
            mrc_link.symlink_to(mrc_dir / f'{ts_name}.mrc')
        if not (mrc_dir / f'{ts_name}.mrc').exists():
            print(f'  WARNING {ts_name}: source .mrc not found at {mrc_dir}')

        # ── Write nodark variant ──────────────────────────────────────────────
        write_trimmed(ts_out / f'{ts_name}_nodark.xf',    xf_lines,    nodark_keep)
        write_tlt    (ts_out / f'{ts_name}_nodark.tlt',   sorted_secs, set(nodark_keep))
        write_order_list(ts_out / f'{ts_name}_nodark_order_list.csv',
                         sorted_secs, set(nodark_keep))
        if xtilt_lines:
            write_trimmed(ts_out / f'{ts_name}_nodark.xtilt', xtilt_lines, nodark_keep)

        write_newst(
            ts_out / 'newst_nodark.com',
            ts_name,
            xf_name      = f'{ts_name}_nodark.xf',
            output_ali   = f'{ts_name}_nodark.ali',
            bin_factor   = args.bin,
            keep_indices = nodark_keep,
        )

        # ── Write clean variant ───────────────────────────────────────────────
        write_trimmed(ts_out / f'{ts_name}_clean.xf',    xf_lines,    clean_keep)
        write_tlt    (ts_out / f'{ts_name}_clean.tlt',   sorted_secs, set(clean_keep))
        write_order_list(ts_out / f'{ts_name}_clean_order_list.csv',
                         sorted_secs, set(clean_keep))
        if xtilt_lines:
            write_trimmed(ts_out / f'{ts_name}_clean.xtilt', xtilt_lines, clean_keep)

        write_newst(
            ts_out / 'newst_clean.com',
            ts_name,
            xf_name      = f'{ts_name}_clean.xf',
            output_ali   = f'{ts_name}_clean.ali',
            bin_factor   = args.bin,
            keep_indices = clean_keep,
        )

        # ── Report ────────────────────────────────────────────────────────────
        n_total      = len(all_sec_idx)
        n_dark       = len(dark_secs)
        n_extra      = len(flagged_secs - dark_secs)
        print(f'{sep}')
        print(f'  {ts_name}')
        print(f'    Total sections         : {n_total}')
        print(f'    Dark excluded          : {n_dark}')
        print(f'    Misaligned (extra)     : {n_extra}')
        print(f'    nodark keeps           : {len(nodark_keep)}  sections')
        print(f'    clean  keeps           : {len(clean_keep)}  sections')
        n_done += 1

    print(sep)
    print(f'\nDone: {n_done} TS processed, {n_skip} skipped')
    print(f'\nFiles written per TS into {out_dir}/ts-xxx_Imod/:')
    print(f'  ts-xxx_nodark.xf / .xtilt / .tlt / _order_list.csv')
    print(f'  ts-xxx_clean.xf  / .xtilt / .tlt / _order_list.csv')
    print(f'  newst_nodark.com  /  newst_clean.com')
    print(f'\nTo run (from inside each ts-xxx_Imod directory):')
    print(f'  submfg newst_nodark.com')
    print(f'  submfg newst_clean.com')


if __name__ == '__main__':
    main()
