"""
Tests for select_ts.py's _compute_ts_stats() fresh_defocus override --
addresses the audit finding that select_ts.py computed reference defocus
from alignment_data.json's own cached per-frame data independently of
shared/parsers.py:compute_reference_defocus (the sanctioned source, reads
ts-*_CTF.txt/_TLT.txt fresh every run), a second, independently-staled
"generation" of the same underlying CTFFIND data that could silently
disagree with it.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands.select_ts import _compute_ts_stats


def _ts_data(frames, thickness_nm=100.0, angpix=1.5, alpha_offset=0.0):
    return {
        'frames': frames,
        'thickness_nm': thickness_nm,
        'angpix': angpix,
        'alpha_offset': alpha_offset,
    }


_FRAMES = [
    {'acq_order': 3, 'tilt': -6.0, 'mean_defocus_um': 3.5, 'rot': 87.0},
    {'acq_order': 1, 'tilt': 0.0, 'mean_defocus_um': 3.0, 'rot': 87.0},
    {'acq_order': 2, 'tilt': -3.0, 'mean_defocus_um': 3.2, 'rot': 87.0},
]


def test_no_fresh_defocus_uses_alignment_data_cached_value():
    """Default (unchanged) behavior: acq_order==1 frame's own
    mean_defocus_um from alignment_data.json."""
    stats = _compute_ts_stats('ts-001', _ts_data(_FRAMES), overlap_thres=None)
    assert stats['ref_defocus_um'] == 3.0


def test_fresh_defocus_overrides_cached_value_when_present():
    fresh = {'ts-001': 9.9}
    stats = _compute_ts_stats('ts-001', _ts_data(_FRAMES), overlap_thres=None,
                              fresh_defocus=fresh)
    assert stats['ref_defocus_um'] == 9.9


def test_fresh_defocus_falls_back_when_ts_not_present():
    """A TS not found in the fresh --input scan (e.g. no CTF.txt for it)
    falls back to alignment_data.json's cached value, not None."""
    fresh = {'ts-999': 9.9}  # different TS
    stats = _compute_ts_stats('ts-001', _ts_data(_FRAMES), overlap_thres=None,
                              fresh_defocus=fresh)
    assert stats['ref_defocus_um'] == 3.0


def test_fresh_defocus_falls_back_when_value_is_none():
    fresh = {'ts-001': None}
    stats = _compute_ts_stats('ts-001', _ts_data(_FRAMES), overlap_thres=None,
                              fresh_defocus=fresh)
    assert stats['ref_defocus_um'] == 3.0


def test_fresh_defocus_applies_even_with_zero_frames():
    """compute_reference_defocus() is independent of alignment_data.json's
    own frames list -- a TS with 0 frames there can still get a fresh
    defocus value."""
    fresh = {'ts-001': 4.4}
    stats = _compute_ts_stats('ts-001', _ts_data([]), overlap_thres=None,
                              fresh_defocus=fresh)
    assert stats['n_frames'] == 0
    assert stats['ref_defocus_um'] == 4.4
