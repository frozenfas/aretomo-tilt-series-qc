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


def mrc_dims(mrc_path):
    """Read (nx, ny, nz) from an MRC header without a mrcfile dependency."""
    with open(mrc_path, 'rb') as f:
        hdr = f.read(12)
    return struct.unpack_from('<3i', hdr, 0)


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
