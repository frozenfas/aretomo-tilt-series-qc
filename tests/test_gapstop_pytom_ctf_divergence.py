"""
Tests for the pytom-match/gapstop-match CTF-missing divergence fix and the
_load_threshold_csv dedup (audit finding: missing CTF data for a single
frame used to fail pytom_match.py's whole TS loudly but gapstop_match.py
inserted a silent NaN defocus for just that frame -- gapstop now fails
that TS loudly too, matching pytom_match.py and the same "diverged twin"
fix already applied once for the missing-_TLT.txt-SEC case).

Synthetic fixtures (no mounted data required) so these always run.
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands import gapstop_match, pytom_match
from aretomo3_preprocess.shared.discovery import load_threshold_csv


# ─────────────────────────────────────────────────────────────────────────────
# _load_threshold_csv dedup
# ─────────────────────────────────────────────────────────────────────────────

def test_both_modules_use_the_same_shared_function():
    assert gapstop_match._load_threshold_csv is load_threshold_csv
    assert pytom_match._load_threshold_csv is load_threshold_csv


def test_load_threshold_csv_basic(tmp_path):
    p = tmp_path / 'thresholds.csv'
    p.write_text('ts_name,threshold\nts-001,3.5\nts-002,4.0\n')
    assert load_threshold_csv(p) == {'ts-001': 3.5, 'ts-002': 4.0}


def test_load_threshold_csv_skips_bad_rows(tmp_path):
    p = tmp_path / 'thresholds.csv'
    p.write_text('ts_name,threshold\nts-001,3.5\nts-002,not_a_number\n')
    assert load_threshold_csv(p) == {'ts-001': 3.5}


# ─────────────────────────────────────────────────────────────────────────────
# gapstop_match._write_wedge_list: fail loud on missing CTF entry, not NaN
# ─────────────────────────────────────────────────────────────────────────────

_FRAMES = [
    {'sec': 1, 'tilt': -3.0},
    {'sec': 2, 'tilt': 0.0},
    {'sec': 3, 'tilt': 3.0},
]


def _args_ns():
    return argparse.Namespace(voltage=300.0, amplitude_contrast=0.1,
                              spherical_aberration=2.7)


def test_write_wedge_list_raises_on_missing_ctf_entry(tmp_path):
    import numpy as np
    # SEC 2 deliberately missing from defocus_df.
    defocus_df = {
        1: {'defocus1_A': 30000.0, 'defocus2_A': 30000.0, 'phase_shift_rad': 0.0},
        3: {'defocus1_A': 31000.0, 'defocus2_A': 31000.0, 'phase_shift_rad': 0.0},
    }
    tilt_angles = np.array([f['tilt'] for f in _FRAMES])
    exposure_arr = np.array([0.0, 4.16, 8.32])

    with pytest.raises(ValueError, match='sec 2'):
        gapstop_match._write_wedge_list(
            tmp_path / 'ts-001_wedgelist.star', 1, 1.5, 100, 100, 50,
            tilt_angles, defocus_df, _FRAMES, exposure_arr, _args_ns())


def test_write_wedge_list_succeeds_when_all_ctf_present(tmp_path):
    import numpy as np
    defocus_df = {
        1: {'defocus1_A': 30000.0, 'defocus2_A': 30000.0, 'phase_shift_rad': 0.0},
        2: {'defocus1_A': 30500.0, 'defocus2_A': 30500.0, 'phase_shift_rad': 0.0},
        3: {'defocus1_A': 31000.0, 'defocus2_A': 31000.0, 'phase_shift_rad': 0.0},
    }
    tilt_angles = np.array([f['tilt'] for f in _FRAMES])
    exposure_arr = np.array([0.0, 4.16, 8.32])

    out_path = gapstop_match._write_wedge_list(
        tmp_path / 'ts-001_wedgelist.star', 1, 1.5, 100, 100, 50,
        tilt_angles, defocus_df, _FRAMES, exposure_arr, _args_ns())
    assert out_path.exists()


def test_write_wedge_list_no_ctf_at_all_still_works(tmp_path):
    """defocus_df=None (no _CTF.txt found at all) is a different,
    already-handled case -- the wedge list is written without a defocus
    column, not treated as an error."""
    import numpy as np
    tilt_angles = np.array([f['tilt'] for f in _FRAMES])
    exposure_arr = np.array([0.0, 4.16, 8.32])

    out_path = gapstop_match._write_wedge_list(
        tmp_path / 'ts-001_wedgelist.star', 1, 1.5, 100, 100, 50,
        tilt_angles, None, _FRAMES, exposure_arr, _args_ns())
    assert out_path.exists()
