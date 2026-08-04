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
    {ts_name: {'coverage_deg', 'high_tilt_ctf_res', 'overall_ctf_res'}} for
    every real TS in an already-loaded alignment_data.json -- coverage is
    the angular range of retained (non-dark) frames; CTF resolution figures
    are mean fit_spacing_A (lower = better), 'high tilt' meaning the top
    tercile of |tilt| for that TS specifically (not a fixed degree cutoff,
    since different TS cover different ranges).
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

        scores[ts_name] = {
            'coverage_deg':      round(coverage_deg, 1),
            'overall_ctf_res_A': round(float(np.mean(res_all)), 2),
            'high_tilt_ctf_res_A': round(float(np.mean(res_high)), 2) if res_high else round(float(np.mean(res_all)), 2),
        }
    return scores


def select_ts(alignment_data: dict, n_ts: int, coverage_pctile: float = 75.0) -> list:
    """
    Pick n_ts TS: keep those with coverage >= coverage_pctile among all TS
    (the "high tilt coverage" qualifying filter), then rank survivors by
    high-tilt CTF resolution ascending (best first), overall CTF resolution
    as tie-breaker. Returns a list of (ts_name, score_dict), best first.
    """
    scores = score_tilt_series(alignment_data)
    if not scores:
        return []
    coverages = [s['coverage_deg'] for s in scores.values()]
    coverage_thresh = float(np.percentile(coverages, coverage_pctile))
    survivors = [(name, s) for name, s in scores.items() if s['coverage_deg'] >= coverage_thresh]
    survivors.sort(key=lambda t: (t[1]['high_tilt_ctf_res_A'], t[1]['overall_ctf_res_A']))
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
