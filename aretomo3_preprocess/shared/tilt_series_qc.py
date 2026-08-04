"""
tilt_series_qc.py — shared TS auto-selection helpers for defocus-handedness
QC (ctf_handedness.py).
"""

import os
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TS auto-selection
# ─────────────────────────────────────────────────────────────────────────────

def score_tilt_series(alignment_data: dict) -> dict:
    """
    {ts_name: {'coverage_deg', 'high_tilt_ctf_res', 'overall_ctf_res',
    'defocus_tilt_spread_um', 'defocus_tilt_slope_um_per_deg'}} for every
    real TS in an already-loaded alignment_data.json -- coverage is the
    angular range of retained (non-dark) frames; CTF resolution figures are
    mean fit_spacing_A (lower = better), 'high tilt' meaning the top
    tercile of |tilt| for that TS specifically (not a fixed degree cutoff,
    since different TS cover different ranges).

    defocus_tilt_spread_um / _slope_um_per_deg describe how flat AreTomo3's
    own overall (not left/right split) per-frame defocus estimate
    (mean_defocus_um) is across tilt. This isn't the handedness signal
    itself -- a genuine left-right defocus gradient partly cancels out of
    the whole-frame average -- it's a proxy for whether *something else*
    (stage/autofocus drift, poor eucentricity) is adding a confounding
    common-mode trend on top of it. A TS where this is flat gives a cleaner
    left/right differential to measure; a TS where it isn't makes any
    handedness read (ours or DefocusGrad's) noisier, since the common-mode
    trend swamps the smaller differential signal on each side.
    """
    scores = {}
    for ts_name, data in alignment_data.items():
        frames = [f for f in data.get('frames', []) if f.get('tilt') is not None]
        if len(frames) < 3:
            continue
        tilts = [f['tilt'] for f in frames]
        coverage_deg = max(tilts) - min(tilts)

        res_all = [f['fit_spacing_A'] for f in frames if f.get('fit_spacing_A') is not None]
        if not res_all:
            continue

        abs_tilts_sorted = sorted(abs(t) for t in tilts)
        high_tilt_cutoff = abs_tilts_sorted[int(len(abs_tilts_sorted) * 2 / 3)]
        res_high = [f['fit_spacing_A'] for f in frames
                    if f.get('fit_spacing_A') is not None and abs(f['tilt']) >= high_tilt_cutoff]

        defocus_pairs = [(f['tilt'], f['mean_defocus_um']) for f in frames
                          if f.get('mean_defocus_um') is not None]
        spread_um = slope_um_per_deg = None
        if len(defocus_pairs) >= 3:
            def_tilts, def_vals = zip(*defocus_pairs)
            spread_um = round(float(np.std(def_vals)), 3)
            slope_um_per_deg = round(float(np.polyfit(def_tilts, def_vals, 1)[0]), 4)

        scores[ts_name] = {
            'coverage_deg':      round(coverage_deg, 1),
            'overall_ctf_res_A': round(float(np.mean(res_all)), 2),
            'high_tilt_ctf_res_A': round(float(np.mean(res_high)), 2) if res_high else round(float(np.mean(res_all)), 2),
            'defocus_tilt_spread_um': spread_um,
            'defocus_tilt_slope_um_per_deg': slope_um_per_deg,
        }
    return scores


def select_ts(alignment_data: dict, n_ts: int, coverage_pctile: float = 75.0) -> list:
    """
    Pick n_ts TS: keep those with coverage >= coverage_pctile among all TS
    (the "high tilt coverage" qualifying filter), then rank survivors by
    how flat AreTomo3's own overall defocus-vs-tilt trend is (ascending
    defocus_tilt_spread_um -- ties towards TS less likely to have a
    confounding drift on top of the handedness signal), high-tilt CTF
    resolution as first tie-breaker, overall CTF resolution as second.
    TS with no mean_defocus_um data (spread_um is None) sort last within
    their coverage tier rather than being dropped. Returns a list of
    (ts_name, score_dict), best first.
    """
    scores = score_tilt_series(alignment_data)
    if not scores:
        return []
    coverages = [s['coverage_deg'] for s in scores.values()]
    coverage_thresh = float(np.percentile(coverages, coverage_pctile))
    survivors = [(name, s) for name, s in scores.items() if s['coverage_deg'] >= coverage_thresh]
    survivors.sort(key=lambda t: (
        t[1]['defocus_tilt_spread_um'] if t[1]['defocus_tilt_spread_um'] is not None else float('inf'),
        t[1]['high_tilt_ctf_res_A'],
        t[1]['overall_ctf_res_A'],
    ))
    return survivors[:n_ts]


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess environment for IMOD tools
# ─────────────────────────────────────────────────────────────────────────────

def imod_env(newstack_bin: str, ctffind_bin: str = None) -> dict:
    """
    Env for subprocesses that shell out to `newstack` (and optionally
    `ctffind`) by bare name -- both binaries' directories need to be on
    PATH, and IMOD_DIR needs to be set for newstack's own wrapper script to
    find its "realbin" -- same pattern as pytom_ribo_auto.py's
    _resample_volume(). ctffind_bin is optional: ctfplotter-based callers
    don't shell out to ctffind at all.
    """
    imod_dir = str(Path(newstack_bin).resolve().parent.parent)  # .../bin/newstack -> ...
    env = dict(os.environ)
    env['IMOD_DIR'] = os.environ.get('IMOD_DIR', imod_dir)
    path_dirs = [str(Path(newstack_bin).parent)]
    if ctffind_bin:
        path_dirs.append(str(Path(ctffind_bin).parent))
    env['PATH'] = f"{':'.join(path_dirs)}:{env.get('PATH', '')}"
    return env
