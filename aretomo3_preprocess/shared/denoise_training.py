"""
Shared helpers for commands that train denoising/dewedging models on
EVN/ODD half-tomogram pairs (cryocare.py, deep_dewedge.py, deep_dewedge_mw.py,
topaz_train.py): finding EVN/ODD pairs, loading per-TS reference defocus from
a ts-select.csv, and defocus-stratified sampling for train/test splits.
"""
import random
from pathlib import Path


def find_evn_odd_pairs(in_dir, vol_suffix):
    """
    Return sorted list of dicts {ts_name, evn, odd} for all EVN/ODD pairs.

    Supports two AreTomo3 naming conventions:
      Single-bin: ts-xxx_EVN_Vol.mrc / ts-xxx_ODD_Vol.mrc
                  (AreTomo3 single --at-bin run; auto-detected when vol_suffix='')
      Multi-bin:  ts-xxx_b4_EVN.mrc / ts-xxx_b4_ODD.mrc
                  (AreTomo3 multiple --at-bin run; use --vol-suffix _b4)

    When vol_suffix='' (default), single-bin volumes (_EVN_Vol.mrc) are tried
    first; falls back to _EVN.mrc if not found.
    """
    in_dir = Path(in_dir)

    def _collect(evn_glob, evn_tag, odd_suffix):
        found = []
        for evn in sorted(in_dir.glob(evn_glob)):
            if not evn.stem.endswith(evn_tag):
                continue
            ts_name = evn.stem[:-len(evn_tag)]
            odd = in_dir / f'{ts_name}{odd_suffix}.mrc'
            if odd.exists():
                found.append({'ts_name': ts_name, 'evn': evn, 'odd': odd})
            else:
                print(f'Warning: {evn.name} has no matching ODD — skipping')
        return found

    # Single-bin volumes: ts-xxx_EVN_Vol.mrc / ts-xxx_ODD_Vol.mrc
    # Try first when vol_suffix is empty or explicitly '_Vol'
    if vol_suffix in ('', '_Vol'):
        pairs = _collect('ts-*_EVN_Vol.mrc', '_EVN_Vol', '_ODD_Vol')
        if pairs:
            return pairs

    # Multi-bin / general pattern: ts-xxx{vol_suffix}_EVN.mrc
    if vol_suffix != '_Vol':
        return _collect(
            f'ts-*{vol_suffix}_EVN.mrc',
            f'{vol_suffix}_EVN',
            f'{vol_suffix}_ODD',
        )

    return []


def load_tsselect_defocus(csv_path):
    """
    Load {ts_name: ref_defocus_um} from a ts-select.csv file.
    Returns empty dict if csv_path is None or the column is absent.
    """
    import csv as _csv
    result = {}
    if csv_path is None:
        return result
    p = Path(csv_path)
    if not p.exists():
        return result
    with open(p, newline='') as fh:
        for row in _csv.DictReader(fh):
            ts  = row.get('ts_name', '').strip()
            val = row.get('ref_defocus_um', '').strip()
            if ts and val:
                try:
                    result[ts] = float(val)
                except ValueError:
                    pass
    return result


def stratified_sample(candidates, n, seed=42):
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
