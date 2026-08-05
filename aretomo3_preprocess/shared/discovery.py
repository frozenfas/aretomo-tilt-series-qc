"""
Shared helpers for the per-TS tomogram-processing commands (membrain-seg,
slabify, pytom-match, gapstop-match, simple-box-mask): finding volumes on
disk, filtering the resulting TS prefixes by --include/--exclude, reading
MRC dimensions, and printing a formatted command.
"""
import re
import struct


def print_cmd(cmd):
    """Print command multi-line, one flag+value per line."""
    it = iter(cmd)
    lines = ['  $ ' + next(it)]
    for tok in it:
        if tok.startswith('-'):
            lines.append('      ' + tok)
        else:
            lines[-1] += '  ' + tok
    print(' \\\n'.join(lines))


def find_volumes(in_dir, vol_suffix=None):
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


def ts_name_from_vol(vol_path, vol_suffix: str) -> str:
    """
    Extract ts_name from a volume Path, stripping vol_suffix and, if
    present, an adjacent _EVN/_ODD denoising-split tag in either order.

    Handles both AreTomo3 naming conventions (vol_suffix='_Vol' by default,
    but may be any user-supplied trailing tag, e.g. '_b4' for a multi-bin
    output with no separate _Vol tag at all):
      ts-001_Vol.mrc      (single-bin main)  -> ts-001
      ts-001_EVN_Vol.mrc  (single-bin EVN)   -> ts-001
      ts-001_ODD_Vol.mrc  (single-bin ODD)   -> ts-001
      ts-001_b4.mrc       (multi-bin main)   -> ts-001
      ts-001_b4_EVN.mrc   (multi-bin EVN)    -> ts-001
      ts-001_b4_ODD.mrc   (multi-bin ODD)    -> ts-001

    Consolidates two byte-for-byte identical copies (imod_mtffilter.py,
    topaz_denoise3d.py). Distinct from find_volumes()'s own internal
    prefix-stripping: that one only ever sees already-EVN/ODD-filtered
    filenames and a narrower vol_suffix meaning (a bin tag inserted before
    a literal '_Vol', not an arbitrary user-supplied trailing tag), so it
    isn't a meaningful duplicate of this.
    """
    stem = vol_path.stem
    for tag in (
        f'_EVN{vol_suffix}',   # e.g. _EVN_Vol  (single-bin EVN)
        f'_ODD{vol_suffix}',   # e.g. _ODD_Vol  (single-bin ODD)
        f'{vol_suffix}_EVN',   # e.g. _b4_EVN   (multi-bin EVN)
        f'{vol_suffix}_ODD',   # e.g. _b4_ODD   (multi-bin ODD)
        vol_suffix,            # e.g. _Vol, _b4 (main volume)
    ):
        if tag and stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC header without a mrcfile dependency."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)


def mrc_pixel_size(mrc_path):
    """
    Read pixel size (Angstrom/px) from an MRC header without a mrcfile
    dependency (cella.x / nx -- nx and mx agree in every file this codebase
    produces, confirmed against real data). Returns None if the header
    can't be read or yields a non-positive size; never raises.

    Consolidates 4 independent implementations that had drifted apart
    (gapstop_match.py's own struct-based reader, plus three separate
    mrcfile-based ones in ctf_handedness.py/imod_mtffilter.py/
    pytom_ribo_auto.py) -- verified numerically identical to mrcfile's own
    voxel_size.x (to float32 rounding) before consolidating.
    """
    with open(mrc_path, 'rb') as f:
        hdr = f.read(1024)
    nx = struct.unpack_from('<i', hdr, 0)[0]
    cell_x = struct.unpack_from('<f', hdr, 40)[0]  # bytes 40-43 = xlen
    if nx > 0 and cell_x > 0:
        return cell_x / nx
    return None


def most_recent_glob(directory, glob_pat):
    """
    Newest-by-mtime match for glob_pat in directory, or None if no match.

    Not just the literal filename (e.g. 'ts_ratings.csv') -- picks up any
    timestamped/renamed copy someone drops in the directory (e.g.
    'ts_ratings_2026-08-01.csv', a common export-tool naming pattern) and
    always prefers the freshest. Consolidates analyse.py's own local
    _most_recent() closure and fixes select_ts.py, which used to check
    only the literal 'ts_ratings.csv' name -- a timestamped ratings
    export showed up correctly in analyse's HTML report but was silently
    ignored by select-ts --select-by-rating, which fell through to
    "treat every TS as unrated" and excluded all of them.
    """
    from pathlib import Path
    matches = sorted(Path(directory).glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_threshold_csv(csv_path):
    """
    Return {ts_name: threshold} from a per-TS threshold override CSV
    produced by the interactive QC report (pytom_match.py/gapstop_match.py
    both consume this -- was a byte-for-byte identical duplicate in each).
    Rows missing 'ts_name'/'threshold' or with a non-numeric threshold are
    silently skipped.
    """
    import csv
    thresholds = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                thresholds[row['ts_name'].strip()] = float(row['threshold'])
            except (KeyError, ValueError):
                pass
    return thresholds


def filter_by_include_exclude(prefixes, include, exclude):
    """
    Filter a list of TS prefixes by --include/--exclude patterns.

    include/exclude: argparse nargs='+' lists -- either one comma-separated
    string or multiple space-separated patterns. '*' is a wildcard.
    """
    if include:
        inc = include[0].split(',') if len(include) == 1 else include
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in inc)]
    if exclude:
        exc = exclude[0].split(',') if len(exclude) == 1 else exclude
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p) for pat in exc)]
    return prefixes
