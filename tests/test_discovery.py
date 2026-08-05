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

from aretomo3_preprocess.shared.discovery import mrc_pixel_size, ts_name_from_vol


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
