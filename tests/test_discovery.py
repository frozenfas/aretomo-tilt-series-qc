"""
Tests for shared/discovery.py's mrc_pixel_size and ts_name_from_vol --
consolidated from 4 (pixel size) and 2 (ts_name) independent duplicate
implementations across gapstop_match.py, ctf_handedness.py,
imod_mtffilter.py, pytom_ribo_auto.py, topaz_denoise3d.py.

Synthetic fixtures (no mounted data required) so these always run.
"""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared.discovery import (
    mrc_pixel_size, ts_name_from_vol, DEFAULT_IN_SKIPS,
)


def _write_fake_mrc_header(path: Path, nx: int, cell_x: float):
    hdr = bytearray(1024)
    struct.pack_into('<i', hdr, 0, nx)
    struct.pack_into('<f', hdr, 40, cell_x)
    path.write_bytes(bytes(hdr))


def test_mrc_pixel_size_basic(tmp_path):
    p = tmp_path / 'ts-001_Vol.mrc'
    _write_fake_mrc_header(p, nx=1022, cell_x=6663.44)
    assert mrc_pixel_size(p) == pytest.approx(6.52, abs=0.01)


def test_mrc_pixel_size_zero_nx_returns_none(tmp_path):
    p = tmp_path / 'bad.mrc'
    _write_fake_mrc_header(p, nx=0, cell_x=100.0)
    assert mrc_pixel_size(p) is None


def test_mrc_pixel_size_zero_cell_returns_none(tmp_path):
    p = tmp_path / 'bad.mrc'
    _write_fake_mrc_header(p, nx=100, cell_x=0.0)
    assert mrc_pixel_size(p) is None


def test_mrc_pixel_size_truncated_file_raises(tmp_path):
    """Unlike a non-positive header value, a file too short to even contain
    a header is a genuine I/O problem -- struct.unpack_from raises, doesn't
    silently return None."""
    p = tmp_path / 'truncated.mrc'
    p.write_bytes(b'\x00' * 10)
    with pytest.raises(struct.error):
        mrc_pixel_size(p)


@pytest.mark.parametrize('filename,vol_suffix,expected', [
    ('ts-001_Vol.mrc', '_Vol', 'ts-001'),
    ('ts-001_EVN_Vol.mrc', '_Vol', 'ts-001'),
    ('ts-001_ODD_Vol.mrc', '_Vol', 'ts-001'),
    ('ts-001_b4.mrc', '_b4', 'ts-001'),
    ('ts-001_b4_EVN.mrc', '_b4', 'ts-001'),
    ('ts-001_b4_ODD.mrc', '_b4', 'ts-001'),
])
def test_ts_name_from_vol(filename, vol_suffix, expected):
    assert ts_name_from_vol(Path(filename), vol_suffix) == expected


def test_ts_name_from_vol_no_match_returns_stem_unchanged():
    assert ts_name_from_vol(Path('ts-001_other.mrc'), '_Vol') == 'ts-001_other'


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_IN_SKIPS -- was three independent hardcoded literals
# (run_aretomo3.py, run_aretomo3_per_ts.py, enrich.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_default_in_skips_value():
    assert set(DEFAULT_IN_SKIPS) == {'_CTF', '_Vol', '_EVN', '_ODD'}


def test_all_three_cli_parsers_share_the_same_in_skips_default():
    import argparse
    from aretomo3_preprocess.commands import run_aretomo3, run_aretomo3_per_ts, enrich

    defaults = {}
    for mod in (run_aretomo3, run_aretomo3_per_ts, enrich):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        mod.add_parser(sub)
        cmd_parser = next(iter(sub.choices.values()))
        action = next(a for a in cmd_parser._actions if a.dest == 'in_skips')
        defaults[mod.__name__] = action.default

    values = list(defaults.values())
    assert all(set(v) == set(DEFAULT_IN_SKIPS) for v in values), defaults


def test_in_skips_defaults_are_independent_list_objects():
    """Each add_argument() call must get its own list copy -- sharing the
    same list object across parsers would let one command's argparse
    mutate another's default."""
    import argparse
    from aretomo3_preprocess.commands import run_aretomo3, enrich

    p1 = argparse.ArgumentParser()
    sub1 = p1.add_subparsers()
    run_aretomo3.add_parser(sub1)
    action1 = next(a for a in next(iter(sub1.choices.values()))._actions
                   if a.dest == 'in_skips')

    p2 = argparse.ArgumentParser()
    sub2 = p2.add_subparsers()
    enrich.add_parser(sub2)
    action2 = next(a for a in next(iter(sub2.choices.values()))._actions
                   if a.dest == 'in_skips')

    assert action1.default is not action2.default
